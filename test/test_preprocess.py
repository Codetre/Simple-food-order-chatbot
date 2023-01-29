from engine.preprocess import Preprocessor

if __name__ == "__main__":
    proc = Preprocessor()

    sentence = "아버지가 방에 들어가신다."
    tp_sequence = proc.token_pos_pairs(sentence)
    pos_sequence = proc.token_or_pos_sequence(sentence, True)

    print(f"원문: {sentence}")
    print(f"토큰+품사: {tp_sequence}")
    print(f"품사만: {pos_sequence}")
