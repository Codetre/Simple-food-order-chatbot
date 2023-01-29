from typing import List
from config import resources as rsc


class Respondent:
    def __init__(self, db, table):
        self._db = db
        self._table = table

    @staticmethod
    def _make_query(intent_name, ner_tags: List, table_name: str):
        select_part = f"SELECT * FROM {table_name}"

        if intent_name and not ner_tags:
            where = f" WHERE intent='{intent_name}' "
        elif intent_name and ner_tags:
            where = f" WHERE intent='{intent_name}' AND bio_tag IN ("
            for tag in ner_tags:
                where += f"'{tag}', "
            where = where[:-2] + ') '

        query = select_part + where
        query += " ORDER BY random() LIMIT 1;"  # 동일한 답변이 2개 이상인 경우, 랜덤으로 선택
        return query

    def search(self, intent_name, ner_tags):
        query = self._make_query(intent_name, ner_tags, self._table)
        # query_tabs = "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
        answer = self._db.fetch(query)

        if not answer:
            query = self._make_query(intent_name, None)
            answer = self._db.fetch(query)

        row_id, intent, tag, ans_txt, rel_url = answer
        return ans_txt, rel_url

    # NER 태그를 실제 입력된 단어로 변환
    @staticmethod
    def tag_to_word(ner_predicts, answer):
        for word, tag in ner_predicts:

            # 변환해야 하는 태그가 있는 경우 추가
            if tag == 'B_FOOD' or tag == 'B_DT' or tag == 'B_TI':
                answer = answer.replace(tag, word)

        answer = answer.replace('{', '')
        answer = answer.replace('}', '')
        return answer
