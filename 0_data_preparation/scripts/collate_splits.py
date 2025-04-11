import argparse
import json
import nltk
from tqdm import tqdm
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_trains", nargs="+", help="train files")
    parser.add_argument("--input_devs", nargs="+", help="dev files")
    parser.add_argument("--input_tests", nargs="+", help="test files")
    parser.add_argument("--output_train", help="Output collated train splits")
    parser.add_argument("--output_dev", help="Output collated dev splits")
    parser.add_argument("--output_tests", help="Output collated test splits")

    args, rest = parser.parse_known_args()
    all_trains = []
    all_tests = []
    all_devs = []

    for train in args.input_trains:
        all_trains += json.load(open(train, "rt"))
    for dev in args.input_devs:
        all_devs += json.load(open(dev, "rt"))
    for test in args.input_tests:
        all_tests += json.load(open(test, "rt"))

    # shuffle 
    random.shuffle(all_trains)
    random.shuffle(all_devs)
    random.shuffle(all_tests)

    print("TOTAL SPLIT SIZES:")
    num_words_train = sum(len(text.split()) for text in all_trains)
    num_words_dev = sum(len(text.split()) for text in all_devs)
    num_words_test = sum(len(text.split()) for text in all_tests)
    print(f"Train: {len(all_trains)} texts, {num_words_train} words")
    print(f"Dev: {len(all_devs)} texts, {num_words_dev} words")
    print(f"Test: {len(all_tests)} texts, {num_words_test} words")

    with open(args.output_train, "wt") as ot:
        for text in all_trains:
            ot.write(text+" ")
    with open(args.output_dev, "wt") as ot:
        for text in all_devs:
            ot.write(text+" ")       
    with open(args.output_tests, "wt") as ot:
        for text in all_tests:
            ot.write(text+" ")
        


    
