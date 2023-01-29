import json
import os.path
import pickle
import threading

from engine import database
from engine.bot import BotServer
from engine.Respondent import Respondent
from engine.model.intent_model import IntentClassifier
from engine.model.ner_model import NamedEntityRecognizer
from engine.preprocess import Preprocessor
from config import resources as rsc
from engine.database import Database


def communicate(sock_conn, addr, params):
    # try:
    #     db = params["db"]
    # except:
    #     conn.close()
    #     print("The database not found.")
    #     return False
    read = sock_conn.recv(2048)
    print("================")
    print(f"Connection from {addr}.")

    if not read:
        print("Connection to the client lost.")
        exit(0)

    recv_data = json.loads(read.decode())
    query = recv_data["query"]

    # 의도 추론.
    intent_name = intent_classifier.predict(query)
    # intent_name = intent_classifier.labels[intent_pred]

    # 개체명 인식.
    named_entity_pred = named_entity_recognizer.predict(query)
    named_entity_tags = named_entity_recognizer.filter_meaningful(named_entity_pred)

    # 의도와 개체명으로 답변 찾기.
    if not os.path.exists(rsc.db):
        print("DB File not found.")
        exit()
    db = Database(rsc.db)
    respondent = Respondent(db, rsc.table)
    try:
        answer_txt, related_img_url = respondent.search(intent_name, named_entity_tags)
        # answer = respondent.tag_to_word(named_entity_pred, answer_text)
    except:
        answer_txt = "이해할 수 없는 질문입니다."
        related_img_url = None

    # 응답으로 답변을 클라이언트에게 전송.
    send_data = {
        "query": query,
        "answer": answer_txt,
        "img_url": related_img_url,
        "intent": intent_name,
        "named_entity": str(named_entity_tags)}
    msg = json.dumps(send_data).encode()
    sock_conn.send(msg)

    # Dealloc conn to DB and client.
    db.close()
    sock_conn.close()


with open(rsc.tokenizer['ner']['word'], 'rb') as fin:  # for ner
    ner_word_tokenizer = pickle.load(fin)
with open(rsc.tokenizer['ner']['tag'], 'rb') as fin:  # for ner
    ner_tag_tokenizer = pickle.load(fin)

# for both of intent, ner
preprocessor = Preprocessor(userdic=rsc.user_dict, enc_dict=rsc.word_index)

# Load the models.
intent_classifier = IntentClassifier(model_dir=rsc.model["intent"],
                                     preprocessor=preprocessor)

named_entity_recognizer = NamedEntityRecognizer(model_dir=rsc.model["ner"],
                                                preprocessor=preprocessor,
                                                word_tokenizer=ner_word_tokenizer,
                                                tag_tokenizer=ner_tag_tokenizer)


def main():
    db = database.Database(rsc.db)

    bot = BotServer(host := rsc.host, port := rsc.port, n_threads := 100)
    bot.create_sock()

    while True:
        conn, cli_addr = bot.ready_for_client()
        params = {}  # "db": db

        # `communicate`에 정의된 일련의 작업에 스레드를 할당한다. 클라이언트 요청마다 새 스레드가 열린다.
        job = threading.Thread(target=communicate, args=(conn, cli_addr, params))
        job.start()

    db.close()


if __name__ == "__main__":
    main()
