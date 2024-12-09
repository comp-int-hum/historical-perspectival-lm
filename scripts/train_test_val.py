import argparse
import json
import random
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Jsonl the structured documents")
    parser.add_argument("--train_portion", help="float of the training portion", default=0.7, type=float)
    parser.add_argument("--dev_portion", help="float of the dev portion", default=0.10, type=float)
    parser.add_argument("--test_portion", help="float of the test portion", default=0.20, type=float)
    parser.add_argument("--output_train", help="output train file", default="data.train")
    parser.add_argument("--output_dev", help="output dev file", default="data.dev")
    parser.add_argument("--output_test", help="output test", default="data.test")
    parser.add_argument("--random_seed", help="random seed", default=42, type=int)
    parser.add_argument("--split_level", help="level to split data at (sentence, paragraph, chapter)", default="sentence")
    args, rest = parser.parse_known_args()

    assert args.split_level in ["sentence", "paragraph", "chapter"], "split level must be one of sentence, paragraph, chapter"

    # setup
    random.seed(args.random_seed)
    logging.basicConfig(level=logging.INFO)


    all_texts = []

    with open(args.input, "rt") as fin:
        for line in fin:
            jline = json.loads(line)
            # each line is a book, with book["structure"] holding a list of chapters, with a list of paragraphs, with a list of sentences
            # the first line is a preface usually so to be safe we skip it
            for i, chapter in enumerate(jline["structure"]):
                if i == 0:
                    continue
                if args.split_level == "sentence":
                    for paragraph in chapter:
                        all_texts.extend(paragraph)
                elif args.split_level == "paragraph":
                    for paragraph in chapter:
                        all_texts.append(" ".join(paragraph))
                elif args.split_level == "chapter":
                    all_texts.append(" ".join([" ".join(paragraph) for paragraph in chapter]))

    logging.info(f"Total sentences: {len(all_texts)}")

    random.shuffle(all_texts)

    train_size = int(len(all_texts) * args.train_portion)
    dev_size = int(len(all_texts) * args.dev_portion)
    test_size = len(all_texts) - train_size - dev_size

    logging.info(f"Train size: {train_size}")
    logging.info(f"Dev size: {dev_size}")
    logging.info(f"Test size: {test_size}")

    with open(args.output_train, "wt") as fout:
        for sentence in all_texts[:train_size]:
            fout.write(sentence + " ")
    
    with open(args.output_dev, "wt") as fout:
        for sentence in all_texts[train_size:train_size+dev_size]:
            fout.write(sentence + " ")

    with open(args.output_test, "wt") as fout:
        for sentence in all_texts[train_size+dev_size:]:
            fout.write(sentence + " ")
    
    logging.info("Done")
    

    
