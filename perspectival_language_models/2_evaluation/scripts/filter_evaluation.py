import argparse
import os
import shutil
import json
import math


def mean(arr):
    return sum(arr) / len(arr)

def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))

def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


# this is a very simple script to start the blimp execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", help="data to filter with")
    parser.add_argument("--evaluation_result", type=str, help="Directory containing the blimp result files")
    parser.add_argument("--min_occurrence", type=int, default=2, help="How often a word must occur to be included")
    parser.add_argument("--output", type=str, help="Filtered results")
    args = parser.parse_args()


    def extract_accuracy(task):
        return task["acc"]
    
    args.evaluation_result = os.path.dirname(args.evaluation_result)
    
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
    common_words = set(word_dictionaries[0].keys())
    for d in word_dictionaries:
        common_words &= set(d.keys())

    min_dict = {}
    for word in common_words:
        min_dict[word] = min(d[word] for d in word_dictionaries)
    word_dictionary = min_dict
    
    
    args.output = os.path.dirname(args.output)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    results_by_category = {}

    words_that_caused_error = {}
    for file in os.listdir(args.evaluation_result):
        if not file.endswith(".jsonl"):
            continue

        task_description = file.replace("_results.jsonl", "")
        results_by_category[task_description] = []
        
        output_file = os.path.join(args.output, file)

        filtered_results = []

        results = json.load(open(os.path.join(args.evaluation_result, file), "r"))
        for task in results:
            if "historical_cloze" in task_description:
                text = " ".join(task["arguments"][0])
            else:
                text = " ".join(task["arguments"][0]) + " " + " ".join(task["arguments"][1])
            text = text.lower()
            cleaned_line = ''.join([i if i.isalpha() or i.isspace() else ' ' for i in text])
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
                filtered_results.append(task)
                if "acc" in task:
                    results_by_category[task_description].append(extract_accuracy(task))
                elif "perplexity" in task:
                    results_by_category[task_description].append(task["perplexity"])

        with open(output_file, "w") as f:
            json.dump(filtered_results, f, indent=2)
    
    # if there is a file ending in _results.json load it
    results_file = [f for f in os.listdir(args.evaluation_result) if f.endswith("_results.json")]
    assert len(results_file) == 1, "There should be exactly one cummulative results file"
    results_file = results_file[0]
    results = json.load(open(os.path.join(args.evaluation_result, results_file), "r"))

    # calculate cummulative results
    if "groups" in results:
        assert len(results["groups"].keys()) == 1, f"We currently only support one group of results"
        cummulative_result_descriptor = list(results["groups"].keys())[0]
        assert len(results_by_category.keys()) == len(results["group_subtasks"][cummulative_result_descriptor]), f"Number of tasks in the group {cummulative_result_descriptor} does not match the number of tasks in the results file. {len(results_by_category.keys())} vs {len(results['group_subtasks'][cummulative_result_descriptor])}"
    #print(words_that_caused_error)
    #print(results_by_category)
    all_accuracies = []
    for accuracies in results_by_category.values():
        all_accuracies.extend(accuracies)
    mean_, stderr = mean(all_accuracies), mean_stderr(all_accuracies)

    if "groups" in results:
        results["groups"][cummulative_result_descriptor]["acc,none"] = mean_
        results["groups"][cummulative_result_descriptor]["acc_stderr,none"] = stderr
        results["results"][cummulative_result_descriptor]["acc,none"] = mean_
        results["results"][cummulative_result_descriptor]["acc_stderr,none"] = stderr

    for task, accuracies in results_by_category.items():
        mean_, stderr = mean(accuracies), mean_stderr(accuracies)
        results["results"][task]["acc,none"] = mean_
        results["results"][task]["acc_stderr,none"] = stderr
    
    # write the results to a file
    with open(os.path.join(args.output, results_file), "w") as f:
        json.dump(results, f, indent=2)

    print(words_that_caused_error)

