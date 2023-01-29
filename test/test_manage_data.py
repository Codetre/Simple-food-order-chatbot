import pickle
from engine.preprocess import Preprocessor

if __name__ == "__main__":
    with open("../data/preprocessor/word_index.bin", "rb") as fin:
        word_index = pickle.load(fin)

    sentence = "내일 오전 10시에 탕수육 주문해줘."

    proc = Preprocessor(userdic="../data/user_dic.tsv")

    token_sequence = proc.token_or_pos_sequence(sentence, True)
    for token in token_sequence:
        matched_index = word_index[token]
        print(f"{token}: {matched_index}")
