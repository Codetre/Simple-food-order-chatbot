#!/usr/bin/env python
# coding: utf-8


import pickle

import tensorflow as tf

from engine.preprocess import Preprocessor


def read_corpus_data(filename, column=1):
    """단어 사전 구축용 말뭉치 데이터를 읽는다.

    :param filename: tsv 파일로 총 4개의 피처로 돼 있다.
    :param column: 추출하기 원하는 열 인덱스.
    :return: 각 줄을 탭 단위로 쪼갠 리스트.
    """
    with open(filename, 'r') as fin:
        lines = [line.split('\t')[column] for line in fin.readlines()]
    return lines


corpus_file = "data/corpus_for_word_dic.tsv"
user_dic_file = "data/user_dic.tsv"
word_index_file = "data/word_index.bin"

corpus = read_corpus_data(corpus_file)

proc = Preprocessor(userdic=user_dic_file)
word_dict = set()  # 말뭉치에서 사용된 단어들 수집.
for sentence in corpus:
    token_sequence = proc.token_or_pos_sequence(sentence, True)
    word_dict = word_dict.union(token_sequence)
print(f"Total {len(word_dict)} tokens collected.")

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(word_dict)
word_index = tokenizer.word_index

with open(word_index_file, "wb") as fout:
    pickle.dump(word_index, fout)
