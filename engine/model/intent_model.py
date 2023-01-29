#!/usr/bin/env python
# coding: utf-8

from functools import reduce

import pandas as pd
import tensorflow as tf

from engine.preprocess import Preprocessor
from config import resources as rsc
from config.hyperparams import intent as intent_hparam


def maxlen_in_sequences(sequences) -> int:
    """시퀀스들 중 가장 긴 것의 길이를 구한다.

    :param sequences: 시퀀스들의 나열.
    :return: 최장 길이.
    """

    def longer_fn(acc, seq):
        return acc if acc > len(seq) else len(seq)

    return reduce(longer_fn, sequences, 0)


def read_file(file):
    """데이터 파일을 읽는다.

    :param file: 1행은 헤더(query,intent)인 CSV 파일을 기대한다.
    :return: 피처별로 분리한 튜플.
        query.shape=(n_samples, len_seq_not_fixed),
        intent.shape=(n_samples,)
    """
    train_df = pd.read_csv(file)
    return train_df["query"], train_df["intent"]


def preprocess(input_feature: pd.Series,
               output_feature: pd.Series,
               preprocessor: Preprocessor):
    """데이터셋 생성.

    :param preprocessor:
    :param input_feature: shape=(n_samples, len_seq_not_fixed)
        각 시퀀스 길이는 일정치 않다.
    :param output_feature: shape=(n_samples,) 피처값은 정수 인코딩된 의도.

    :return: trainset, evalset, testset
    """
    enc_sequences = []

    data_size = len(input_feature)
    train_size = int(data_size * intent_hparam["train_rate"])
    eval_size = int(data_size * intent_hparam["eval_rate"])
    test_size = data_size - (train_size + eval_size)

    for query in input_feature:
        token_sequence = preprocessor.token_or_pos_sequence(query, True)
        enc_sequence = preprocessor.vectorize(token_sequence)
        enc_sequences.append(enc_sequence)
    maxlen = maxlen_in_sequences(enc_sequences)

    padded_vectors = tf.keras.preprocessing.sequence.pad_sequences(
        enc_sequences, maxlen=maxlen, padding="post")

    dataset = tf.data.Dataset.from_tensor_slices((padded_vectors, output_feature))
    dataset = dataset.shuffle(data_size)

    trainset = dataset.take(train_size).batch(intent_hparam["batch_size"])
    evalset = dataset.skip(train_size).take(eval_size).batch(intent_hparam["batch_size"])
    testset = dataset.skip(train_size + eval_size).take(test_size).batch(intent_hparam["batch_size"])

    return trainset, evalset, testset


def build_model(n_timesteps,
                output_size,
                vocab_size,
                emb_size,
                dropout_rate):
    """의도 분류 모델 빌드.

    input_shape = (None, n_timesteps), output_shape = (output_size,)

    :param n_timesteps: 시퀀스의 길이.
    :param output_size: 가능한 의도의 수.
    :param vocab_size: 단어 임베딩의 단어 공간 크기.
    :param emb_size: Embedding 레이어가 만든 임베딩 벡터 길이.
    :param dropout_rate: 드롭아웃 비율.
    :return: 컴파일된 모델.
    """
    # Stack layers.
    input_layer = tf.keras.layers.Input(shape=(n_timesteps,))
    # Embedding.input_shape = (batch_size, input_length)
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=emb_size,
        input_length=n_timesteps)(input_layer)
    # Embedding.output_shape = (batch_size, input_length, output_dim)
    dropout = tf.keras.layers.Dropout(rate=dropout_rate)(embedding)

    # Conv1D.input_shape = (samples, time, features)
    conv1 = tf.keras.layers.Conv1D(
        filters=emb_size,
        kernel_size=3,
        activation=tf.nn.relu)(dropout)
    pool1 = tf.keras.layers.GlobalMaxPool1D()(conv1)

    conv2 = tf.keras.layers.Conv1D(
        filters=emb_size,
        kernel_size=4,
        activation=tf.nn.relu)(dropout)
    pool2 = tf.keras.layers.GlobalMaxPool1D()(conv2)

    conv3 = tf.keras.layers.Conv1D(
        filters=emb_size,
        kernel_size=5,
        activation=tf.nn.relu)(dropout)
    pool3 = tf.keras.layers.GlobalMaxPool1D()(conv3)

    concat = tf.keras.layers.concatenate([pool1, pool2, pool3])

    dense = tf.keras.layers.Dense(
        emb_size, activation=tf.nn.relu)(concat)
    dropout2 = tf.keras.layers.Dropout(dropout_rate)(dense)

    logits = tf.keras.layers.Dense(output_size)(dropout2)
    pred = tf.keras.layers.Dense(output_size, activation=tf.nn.softmax)(logits)

    model = tf.keras.Model(inputs=input_layer, outputs=pred)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model


class IntentClassifier:
    def __init__(self, model_dir: str,
                 preprocessor: Preprocessor):
        self.labels = {0: "인사", 1: " 욕설", 2: "주문", 3: "예약", 4: "기타"}  # preprocessor.index_word
        self._model = tf.keras.models.load_model(model_dir)
        print("The kernel intent model loaded.")
        self.preprocessor = preprocessor

    def predict(self, sentence: str) -> str:
        """텍스트를 받아 그 의도를 출력한다.

        입력 전처리는 1. 토큰 분리 2. 인코딩 3. 패딩 순서로 이뤄진다.

        :param sentence: 텍스트(잠정적으로 한 문장).
        :return: 디코딩된 의도.
        """
        maxlen = self._model.input_spec[0].shape[1]
        # 입력 전처리.
        token_sequence = self.preprocessor.token_or_pos_sequence(sentence, True)
        vector = self.preprocessor.vectorize(token_sequence)
        # 메서드 `pad_sequences`는 shape=(n_sequences)인 인자를 받으므로 리스트로 벡터를 감쌌다.
        padded_vectors = tf.keras.preprocessing.sequence.pad_sequences(
            [vector], maxlen=maxlen, padding="post")

        # shape=(n_seqs, intent_space_size)이며, shape[1]은 softmax 확률값이다.
        preds = self._model.predict(padded_vectors)

        most_likely = tf.math.argmax(preds, axis=1)
        encoded_intent = most_likely.numpy()[0]  # 실제 샘플은 하나 뿐이니 [0]으로 참조.
        return self.labels[encoded_intent]  # 읽을 수 있는 문자열로 치환.


def main():
    # Prepare dataset.
    preprocessor = Preprocessor(enc_dict=rsc.word_index, userdic=rsc.user_dict)
    queries, intents = read_file(rsc.train["intent"])

    vocab_size = len(preprocessor.word_index) + 1
    output_size = len(set(intents.values))
    trainset, evalset, testset = preprocess(queries, intents, preprocessor)

    # Build model.
    model = build_model(n_timesteps=maxlen,
                        output_size=output_size,
                        emb_size=intent_hparam["emb_size"],
                        vocab_size=vocab_size,
                        dropout_rate=intent_hparam["dropout_rate"])
    print(model.summary())

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=5, mode="auto")
    history = model.fit(trainset,
                        validation_data=evalset,
                        epochs=intent_hparam["epochs"],
                        callbacks=[early_stopping])

    loss, acc = model.evaluate(testset)
    print(f"loss: {loss}, acc: {acc}")

    model.save(rsc.model["intent"])


if __name__ == "__main__":
    main()
