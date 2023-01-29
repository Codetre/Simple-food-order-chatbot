from itertools import zip_longest
import pickle

import numpy as np
from seqeval.metrics import f1_score, classification_report
import tensorflow as tf

from config import resources as rsc
from config import hyperparams as hparams
from engine.preprocess import Preprocessor


def read_file(file):
    """데이터 파일을 파싱

    파일은
        1. 세미콜론으로 시작하는 원본 문장,
        2. 그 다음엔 달러 기호로 시작하는 BIO 태깅된 문장,
        3. 그 이후부터 다음 원본 문장 전까지는 각 토큰의 피처들의 나열(ID, token, POS, BIO)
    이렇게 세 종류의 줄로 이루어져 있다.

    :param file: BIO 토큰 정보를 가진 파일.
    :return: 각 줄의 모든 피처를 열거한 리스트: [feats_sentence1, feats_sentence2, ...]
    """

    # 어떤 라인인지 파악하기 위한 조건
    def is_original_line(cur_i, lines):
        if cur_i >= len(lines) - 1:
            return False
        else:
            cur_line, next_line = lines[cur_i], lines[cur_i + 1]
            cur_first_char = cur_line[0]
            next_first_char = next_line[0]
            return cur_first_char == ";" and next_first_char == '$'

    def is_ner_processed_line(cur_i, lines):
        cur_line = lines[cur_i]
        first_char = cur_line[0]
        return first_char == "$"

    def is_end_of_datapoint(line):
        first_char = line[0]
        return first_char == "\n"

    features_all_lines = []  # 파일 내 모든 문장의 토큰 피처들
    with open(file, 'r', encoding="utf-8") as fin:
        lines = fin.readlines()

        for i, line in enumerate(lines):
            if is_original_line(i, lines):
                features_a_line = []
            elif is_ner_processed_line(i, lines):
                continue
            elif is_end_of_datapoint(line):
                features_all_lines.append(features_a_line)
            else:
                features_a_line.append(line.split())
    return features_all_lines


def init_tokenizer(sequences, **opts):
    """훈련시킨 토크나이저 인스턴스를 내보낸다.

    :param sequences: 토크나이저 훈련시킬 시퀀스들.
    :param opts: 토크나이저 초기화 시 줄 옵션.
        - words_all_sentences면 `oov_token="OOV"`
        - tags_all_sentences면 `lower=False` 부여.

    :return: tokenizer.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(**opts)
    tokenizer.fit_on_texts(sequences)
    tokenizer.index_word[0] = 'PAD'
    return tokenizer


def encode_sequences(sequences, tokenizer):
    """문자열 시퀀스를 정수형으로 인코딩.

    :param sequences: words_all_sentences와 tags_all_sentences 두 가지 중 하나.
    :param tokenizer: 미리 훈련된 케라스 토크나이저 인스턴스.
    :return: 토크나이저에 의해 인코딩된 시퀀스. 인코딩에 사용한 토크나이저, 최장 시퀀스 길이 세 가지.
    """
    encoded_sequences = tokenizer.texts_to_sequences(sequences)
    maxlen = max(map(lambda sequence: len(sequence), encoded_sequences))

    return encoded_sequences, maxlen


def divide_words_tags(corpus):
    """tokens, tags 따로 수집.

    샘플별로 token과 BIO 태그를 분리한다.

    :param corpus: `read_file`의 리턴.
    :return: (tokens, bio_tags)
    """
    words_all_sentences, tags_all_sentences = [], []
    for info_about_a_sentence in corpus:
        words_a_sentence, tags_a_sentence = [], []

        for features_a_token in info_about_a_sentence:
            word, tag = features_a_token[1], features_a_token[3]
            words_a_sentence.append(word)
            tags_a_sentence.append(tag)

        words_all_sentences.append(words_a_sentence)
        tags_all_sentences.append(tags_a_sentence)
    return words_all_sentences, tags_all_sentences


def enc_pad_sequences(tokens_all_sentences, tokenizer, maxlen=None):
    enc_word_sequences, t_maxlen = encode_sequences(tokens_all_sentences, tokenizer)
    if not maxlen:
        maxlen = t_maxlen

    padded_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        enc_word_sequences, maxlen=maxlen, padding="post")
    return padded_word_sequences, maxlen


def preprocess(words_all_sentences, word_tokenizer, tags_all_sentences, tag_tokenizer):
    """데이터 전처리 후 데이터셋 생성.

    :param words_all_sentences: divide_words_tags의 첫번째 리턴
    :param word_tokenizer: 훈련된 단어 토크나이저.
    :param tags_all_sentences: divide_words_tags의 두번째 리턴
    :param tag_tokenizer: 훈련된 태그 토크나이저.
    :return: dataset, maxlen.
        - dataset: SliceDataset.
            [0]: 인코딩된 고정 길이 토큰 시퀀스,
            [1]: 원-핫 태그. 토큰 시퀀스가 어느 태그에 속하는지 의미.
        - maxlen: 토큰 시퀀스 길이.
    """
    # enc_word_sequences, maxlen = encode_sequences(words_all_sentences, word_tokenizer)
    # padded_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    #     enc_word_sequences, maxlen=maxlen, padding="post")

    # enc_tag_sequences, _ = encode_sequences(tags_all_sentences, tag_tokenizer)
    # padded_tag_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    #     enc_tag_sequences, maxlen=maxlen, padding="post")

    padded_word_sequences, maxlen = enc_pad_sequences(words_all_sentences, word_tokenizer)
    padded_tag_sequences, _ = enc_pad_sequences(tags_all_sentences, tag_tokenizer, maxlen)

    one_hot_tags = tf.keras.utils.to_categorical(
        padded_tag_sequences, num_classes=len(tag_tokenizer.index_word) + 1)
    dataset = tf.data.Dataset.from_tensor_slices(
        (padded_word_sequences, one_hot_tags))

    return dataset, maxlen


def build_model(input_len,
                vocab_size: int,
                tag_size: int):
    """NER 모델을 빌드한다.

    입력 shape = (len_seq,). 정수형 인코딩된 토큰 시퀀스를 하나 받는다.
    출력 shape = (tag_space_size,). 토큰 시퀀스가 각 BIO 태그일 확률을 나타낸다.

    :param input_len: 입력 토큰 시퀀스의 길이. 입력된느 모든 시퀀스의 길이는 일정해야 한다.
    :param vocab_size: 토큰 공간의 크기.
    :param tag_size: BIO 태그 공간의 크기.
    :return: 컴파일된 모델.
    """
    input = tf.keras.layers.Input(shape=[input_len, ])
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=30,
        mask_zero=True,  # 시퀀스를 0으로 패딩했다면 True로 값을 설정해 그 0이 패딩임을 고지.
        input_length=input_len)(input)
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            200,
            return_sequences=True,
            dropout=.5,
            # T면 [timesteps, batch, feature] F면 [batch, timesteps, feature]의 입력
            # 형상으로 True일 떄가 연산 효율은 더 좋다(기본값 False).
            # time_major= True
            recurrent_dropout=.25))(embedding)
    dense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(tag_size, activation="softmax"))(bi_lstm)

    bi_lstm_model = tf.keras.Model(inputs=input, outputs=dense)
    bi_lstm_model.compile(loss="categorical_crossentropy",
                          optimizer=tf.keras.optimizers.legacy.Adam(),
                          metrics=["accuracy"])

    return bi_lstm_model


def sequences_to_tag(sequences, mapping_dict):
    """"one-hot 결과를 읽을 수 있는 문자로 변환한다.

    각 시퀀스는 토큰별 one-hot 추론 결과를 원소로 담고 있다. [num_seq, len_seq, len_one_hot]

    :param sequences: 추론된 원-핫 형식 태그 콜렉션. 추론 함수의 결과 텐서를 기대한다.
    :param mapping_dict: 인코딩된 태그값에 해당하는 원래 문자 태그의 매핑.
    :return: `mapping_dict`에 의해 문자로 변환된 시퀀스들.
    """
    tags_all_sequences = []
    for sequence in sequences:
        tags_a_sequence = []
        for one_hot_score in sequence:
            most_likely_index = np.argmax(one_hot_score)
            pred_tag = mapping_dict[most_likely_index].replace("PAD", "O")
            tags_a_sequence.append(pred_tag)
        tags_all_sequences.append(tags_a_sequence)

    return tags_all_sequences


class NamedEntityRecognizer:
    """NER(named-entity recognition) 모델 작동을 위한 클래스.

    주요 기능: 텍스트를 입력 받아 여기서 개체명 하나를 찾아 돌려준다.
    이 클래스는 챗봇 엔진이 기동될 때 훈련된 모델 파일을 읽음으로써 기동된다.
    """

    def __init__(self,
                 model_dir: str,
                 preprocessor: Preprocessor,
                 word_tokenizer: tf.keras.preprocessing.text.Tokenizer,
                 tag_tokenizer: tf.keras.preprocessing.text.Tokenizer):
        self._model = tf.keras.models.load_model(model_dir)
        print("The kernel ner model loaded.")
        self.preprocessor = preprocessor
        self.word_tokenizer = word_tokenizer
        self.tag_tokenizer = tag_tokenizer

    def predict(self, query: str):
        def find_maxlen_in_embedding_layer():
            """현재 모델의 임베딩 레이어의 maxlen을 찾기 위한 함수.

            :return: self.model의 임베딩 레이어.
            """
            for layer in self._model.layers:
                if isinstance(layer, tf.keras.layers.Embedding):
                    return layer.input_length

        tokens = self.preprocessor.token_or_pos_sequence(query, only_token=True)
        enc_word_sequences, _ = encode_sequences([tokens],
                                                 self.word_tokenizer)
        maxlen = find_maxlen_in_embedding_layer()  # 훈련 시의 고정 크기를 알아내야 한다.
        padded_word_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            enc_word_sequences, maxlen=maxlen, padding="post")

        # maxlen = find_maxlen_in_embedding_layer()
        # sequences = [self.preprocessor.vectorize(tokens)]
        preds = self._model.predict(np.array(padded_word_sequences))

        # 리턴값을 위해 인코딩되지 않았으면서, 패딩된 시퀀스 만들기.
        words = ['OOV'] * maxlen
        for i, token in enumerate(tokens):
            words[i] = token

        tags = sequences_to_tag(preds, self.tag_tokenizer.index_word)
        return list(zip(words, *tags))

    @staticmethod
    def filter_meaningful(token_tag_list):
        """의미 있는 NER 태그만 뽑아낸다.

        :param token_tag_list: `predict`의 리턴값.
        :return: 의미 있는 태그만 수집.
        """
        return [tag for token, tag in token_tag_list if tag != 'O']
        # tags = []
        # for _, tag in token_tag_list:
        #     if tag != 'O':
        #         tags.append(tag)
        # if len(tags) == 0:
        #     return None
        # return tags


def main():
    # Build a dataset.
    corpus = read_file(rsc.train["ner"])
    words_all_queries, tags_all_queries = divide_words_tags(corpus)
    query_tokenizer = init_tokenizer(words_all_queries, oov_token="OOV")
    bio_tag_tokenizer = init_tokenizer(tags_all_queries, lower=False)
    trainset, maxlen = preprocess(words_all_queries, query_tokenizer,
                                  tags_all_queries, bio_tag_tokenizer)

    # 엔진에 들어오는 입력을 인코딩하기 위해 토크나이저 둘 다 파일로 보존.
    with open(rsc.tokenizer["ner"]["word"], "wb") as fout:
        pickle.dump(query_tokenizer, fout)

    with open(rsc.tokenizer["ner"]["tag"], "wb") as fout:
        pickle.dump(bio_tag_tokenizer, fout)

    print(f"Tokenizers(maxlen={maxlen}) saved.")

    # 토크나이저의 인코딩은 0번을 OOV를 위해 할당하는 점을 잊지 말 것.
    word_size = len(query_tokenizer.index_word) + 1
    tag_size = len(query_tokenizer.index_word) + 1
    trainset = trainset.batch(hparams.ner["batch_size"])

    # Build and train a model.
    bi_lstm_model = build_model(input_len=maxlen,
                                vocab_size=word_size,
                                tag_size=tag_size)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                      patience=5,
                                                      mode="auto")
    bi_lstm_history = bi_lstm_model.fit(trainset,
                                        epochs=hparams.ner["epochs"],
                                        callbacks=[early_stopping])
    bi_lstm_model.save(rsc.model["ner"])

    # Model validation.
    test_corpus = read_file(rsc.test["ner"])
    test_words_all_queries, test_tags_all_queries = divide_words_tags(test_corpus)

    test_x = query_tokenizer.texts_to_sequences(test_words_all_queries)
    test_y = bio_tag_tokenizer.texts_to_sequences(test_tags_all_queries)
    pred_y = bi_lstm_model.predict(test_x)

    decoded_pred_tags = sequences_to_tag(pred_y, bio_tag_tokenizer.index_word)
    decoded_test_tags = sequences_to_tag(test_y, bio_tag_tokenizer.index_word)

    report = classification_report(decoded_test_tags, decoded_pred_tags)
    f1 = f1_score(decoded_test_tags, decoded_pred_tags)

    print(report)
    print(f1)


if __name__ == "__main__":
    main()
