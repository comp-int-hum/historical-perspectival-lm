import argparse
import subprocess
import os
import shutil
import json

# this is a very simple script to start the blimp execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", help="data to filter with")
    parser.add_argument("--blimp_dir", type=str, help="Directory containing the blimp files")
    parser.add_argument("--min_occurrence", type=int, default=2, help="How often a word must occur to be included")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()

    word_dictionaries = []
    for file in args.data:
        with open(file, "r") as f:
            text_data = f.read()
    
        word_dictionary = {}
        for line in text_data.split("\n"):
            # remove any non alphabetic characters
            line = ''.join([i if i.isalpha() or i.isspace() else ' ' for i in line]).lower()
            for word in line.split(" "):
                if len(word) == 0:
                    continue
                if word not in word_dictionary:
                    word_dictionary[word] = 0
                word_dictionary[word] += 1
        word_dictionaries.append(word_dictionary)
    
    # merge the word dictionaries so that the min count is kept
    min_word_dictionary = word_dictionaries[0]
    for word_dictionary in word_dictionaries:
        for word, count in word_dictionary.items():
            if word not in min_word_dictionary:
                min_word_dictionary[word] = 0
            min_word_dictionary[word] = min(min_word_dictionary[word], count)
    word_dictionary = min_word_dictionary
    

    args.output = os.path.dirname(args.output)
    # clear the output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)

    os.makedirs(args.output, exist_ok=True)

    words_that_caused_error = {}
    for file in os.listdir(args.blimp_dir):
        # file is a jsonl file
        for line in open(os.path.join(args.blimp_dir, file), "r"):
            line = json.loads(line)
            text_line = (line["sentence_good"] + " " + line["sentence_bad"]).lower()
            cleaned_line = ''.join([i if i.isalpha() or i.isspace() else ' ' for i in text_line])
            words = cleaned_line.split(" ")
            valid_line = True
            for word in words:
                if len(word) == 0:
                    continue
                if word not in word_dictionary or word_dictionary[word] <= args.min_occurrence:
                    valid_line = False
                    if word not in words_that_caused_error:
                        words_that_caused_error[word] = 0
                    words_that_caused_error[word] += 1
            if valid_line:
                with open(os.path.join(args.output, file), "a") as f:
                    f.write(json.dumps(line) + "\n")
    print(words_that_caused_error)

