import argparse

import engine.model.ner_model as ner
import engine.model.intent_model as intent
from engine import mainloop as server

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='for main module runner')
parser.add_argument('-m', '--mode')  # option that takes a value
args = parser.parse_args()

print(args.mode)

if __name__ == '__main__':
    if args.mode == 'train_ner':
        ner.main()
    elif args.mode == 'train_intent':
        intent.main()
    elif args.mode == 'run_server':
        server.main()
