#!/usr/bin/env python
# coding: utf-8
from functools import partial
from typing import List, Dict
import pickle

from konlpy.tag._komoran import Komoran

from config import resources as rsc


class Preprocessor:
    """텍스트를 토큰 및 형태소 단위로 쪼개는 형태소 분석기를 핵심 기능으로 하는 클래스."""

    def __init__(self, enc_dict: str = None, userdic=None):
        """

        :param enc_dict: `Dict[str, int]` 형식의 직렬화된 데이터를 담고 있는 파일 경로.
        :param userdic: 형태소 분석기에게 알려줄 사용자 사전 경로.
        """
        self._morph_analyzer = Komoran(userdic=userdic)
        if enc_dict:
            with open(enc_dict, "rb") as fin:
                self.word_index = pickle.load(fin)
                self.index_word = {v: k for k, v in self.word_index.items()}
        else:
            self.word_index = None

        self.pos_to_exclude = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    def token_pos_pairs(self, sentence: str):
        """문장 속 토큰별로 품사를 태깅한다.

        :param sentence: 형태소 분석할 문장.
        :return: shape = (n_tokens, 2) shape[1] = (token, pos)
        """
        return self._morph_analyzer.pos(sentence)

    def token_or_pos_sequence(self, sentence: str, only_token=False):
        """토큰 혹은 형태소만으로 이루어진 시퀀스 생성.

        `token_pos_pairs`가 반환하는 (토큰, 품사) 리스트에서, 토큰 혹은 품사 중 하나만으로
        이루어진 시퀀스를 뽑아낸다.

        :param sentence: 분석할 문장.
        :param only_token: 결과에 품사를 넣을지 결정.
        :return: 토큰 혹은 품사 시퀀스. shape = (n_tokens,)
        """

        def pos_exclude(pos):
            return pos in self.pos_to_exclude

        token_pos_sequence = self.token_pos_pairs(sentence)
        sequence = []
        for token, pos in token_pos_sequence:
            if not pos_exclude(pos):
                sequence.append(pos if not only_token else token)
        return sequence

    def vectorize(self, sequence: List[str]) -> List[int]:
        """문자 시퀀스를 임베딩 벡터로 변환.

        :param sequence: 단어 시퀀스. shape=(n_tokens,)
        :return: `word_index`에 기반해 인코딩된 벡터.
            `word_index`가 있다면 shape = (n_tokens,), 없다면 빈 벡터.
        """
        if not self.word_index:
            print("Cannot find a file with the matching index for a word.")
            return []

        def match(word_index: Dict[str, int], token: str):
            """단어-인덱스 매핑에서 단어에 해당하는 인덱스를 찾는다.

            :param word_index: 키: 토큰, 값: 정수
            :param token: 단어.
            :return: word_index[token] or word_index["OOV"]
            """
            if value := word_index.get(token):
                index = value
            else:
                index = word_index["OOV"]
            return index

        vector = list(map(partial(match, self.word_index), sequence))
        return vector


if __name__ == "__main__":
    preprocessor = Preprocessor(rsc.word_index, rsc.user_dict)
    sequence = preprocessor.token_or_pos_sequence("나랑 만날래?", True)
    result = preprocessor.vectorize(sequence)
