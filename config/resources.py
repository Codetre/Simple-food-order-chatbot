import os
import sys

ROOT_DIR = os.path.abspath(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(ROOT_DIR)

user_dict = os.path.join(DATA_DIR, 'preprocessor', 'user_dic.tsv')
word_index = os.path.join(DATA_DIR, 'preprocessor', 'word_index.bin')

train = {
    "intent": os.path.join(DATA_DIR, 'training', 'total_train_data.csv'),
    "ner": os.path.join(DATA_DIR, 'model', 'training', 'ner_train.txt')
}

test = {
    "intent": "",
    "ner": os.path.join(DATA_DIR, "model', 'test', 'ner_test.txt")
}

tokenizer = {
    "ner": {
        "word": os.path.join(DATA_DIR, 'engine', 'ner_word_tokenizer.pickle'),
        "tag": os.path.join(DATA_DIR, 'engine', 'ner_tag_tokenizer.pickle'),
    }
}

model = {
    "ner": os.path.join(ROOT_DIR, 'models', 'engine', 'named_entity_recognizer'),
    "intent": os.path.join(ROOT_DIR, 'models', 'engine', 'intent_classifier')
}

db = os.path.join(DATA_DIR, "engine", "chatbot.sqlite")
table = 'response'
host = '0.0.0.0'  # 로컬에서 접속 시 127.0.0.1가 된다.
port = 5050
