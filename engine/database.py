from typing import List, Tuple
import sqlite3

import pandas as pd


class Database:
    def __init__(self, path: str):
        """DB 래퍼 인스턴스 초기화

        :param path: SQLite 데이터베이스 파일 경로.
        """
        self.path = path
        self.conn = sqlite3.connect(path)
        print("Establish DB connection.")

    # def connect(self):
    #     if self.conn:
    #         print("Already connected.")
    #     else:
    #         self.conn = sqlite3.connect(self.path)
    #         self.cursor = self.conn.cursor()

    def close(self):
        if not self.conn:
            print("Already disconnected.")
        else:
            self.conn.close()

    def execute(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        self.conn.commit()
        last_row_id = cursor.lastrowid
        return last_row_id

    def fetch(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        cursor.close()
        return result

    def fetch_all(self, query):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result


class Manager:
    def __init__(self, db: Database):
        self.db = db

    def _make_query(self, intent_name, named_entity_tags):
        pass

    def search(self, intent_name, named_entity_tags):
        pass

    def tag_to_word(self, named_entity_preds, answer):
        pass


def clear_table(db_conn, table_name):
    """테이블 내 데이터를 모두 지운다.

    :param db_conn: 접속된 DB 연결.
    :param table_name: 비우고자 하는 테이블명.
    :return: 제거 됐는지 여부.
    """
    cursor = db_conn.cursor()

    result = cursor.execute(f"DELETE FROM {table_name};")
    db_conn.commit()
    rows = result.fetchall()

    cursor.close()
    return True if not rows else False


def insert_data(db_conn: sqlite3.Connection, datapoints: List[str]):
    """

    :param db_conn: 접속된 DB 연결.
    :param datapoints: 사용자가 입력하는 학습 데이터 파일에서 가져온 한 행.
        피처들로는 intent, ner, query, answer, img_url가 있다.
    :return:
    """
    intent, ner, query, answer, img_url = datapoints
    query = f"""INSERT INTO train_data (
    intent, ner, query, answer, img_url)
    VALUES ('{intent}', '{ner}', '{query}', '{answer}', '{img_url}');"""
    print(query)

    cursor = db_conn.cursor()
    result = cursor.execute(query)
    db_conn.commit()

    cursor.close()
    return result


def parse_row(series: pd.Series) -> Tuple[str]:
    """데이터 한 줄을 INSERT 쿼리를 위해 처리.

    :param series: 사용자 입력 챗봇 데이터 파일을 데이터프레임으로 읽은 한 줄.
    :return: intent, ner, query, answer, img_url 순.
    """
    return series[1].fillna("null").values
