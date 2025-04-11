import argparse
import json
import nltk
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_trains", nargs="+", help="train files")
    parser.add_argument("--input_devs", nargs="+", help="dev files")
    parser.add_argument("--input_tests", nargs="+", help="test files")
    parser.add_argument("--output_train", help="Output collated train splits")
    parser.add_argument("--output_dev", help="Output collated dev splits")
    parser.add_argument("--output_tests", help="Output collated test splits")

    args, rest = parser.parse_known_args()

    with open(args.output_train, "wt") as ot:
        for it in args.input_trains:
            with open(it, "rt") as t_in:
                ot.write(t_in.read()+" ")
    with open(args.output_dev, "wt") as ot:
        for it in args.input_devs:
            with open(it, "rt") as t_in:
                ot.write(t_in.read()+" ")        
    with open(args.output_tests, "wt") as ot:
        for it in args.input_tests:
            with open(it, "rt") as t_in:
                ot.write(t_in.read()+" ")
        


    
