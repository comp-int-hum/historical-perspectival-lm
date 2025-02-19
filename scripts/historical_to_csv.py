import argparse
import subprocess
import os
import json
import pandas as pd
from tqdm import tqdm
import glob


# extract data from normal blimp file
def extract_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    flattened_rows = []
    for entry in data:
        doc = entry.get("doc", {})
        response = entry.get("resps", False)
        row = {
            #"doc_id": entry.get("doc_id"),
            "sentence_one": doc.get("sentence_one"),
            "sentence_two": doc.get("sentence_two"),
            "more_likely_sentence": entry.get("more_likely_sentence"),
            #"field": doc.get("field"),
            #"linguistics_term": doc.get("linguistics_term"),
            #"UID": doc.get("UID"),
            #"simple_LM_method": doc.get("simple_LM_method"),
            #"one_prefix_method": doc.get("one_prefix_method"),
            #"two_prefix_method": doc.get("two_prefix_method"),
            #"lexically_identical": doc.get("lexically_identical"),
            #"pair_id": doc.get("pairID") if "pairID" in doc else doc.get("pair_id"),
            #"target": entry.get("target"),
            #"acc": entry.get("acc"),
            # Store nested structures as strings if needed
            #"arguments": json.dumps(entry.get("arguments", [])),
            "perplexity_one": response[0][0][0],
            "perplexity_two": response[1][0][0],
            #"response_one": response[0][0][1],
            #"response_two": response[1][0][1],
        }
        flattened_rows.append(row)
    return flattened_rows

# extract data from results file
def extract_data_results(file):
    with open(file, 'r') as f:
        data = json.load(f)
    flattened_rows = []
    for entry, values in data.items():
        row = {
            "acc" : values["acc,none"],
            "acc_stderr" : values["acc_stderr,none"],
            "alias" : values["alias"],
        }
        flattened_rows.append(row)
    return flattened_rows


def merge_dataframes(dfs, prefixes):
    assert len(dfs) == len(prefixes), "Number of dataframes must match number of prefixes"
    all_columns = set()

    all_values_to_merge_on = {}
    for df in dfs:
        all_columns.update(df.columns)

    common_columns = set()
    
    final_df = dfs[0].copy()
    
    for column in all_columns:
        same_in_all = True
        for df in dfs:
            if column not in df.columns:
                same_in_all = False
                break
            if not df.get(column).equals(final_df.get(column)):
                same_in_all = False
                break
        if same_in_all:
            common_columns.add(column)

    common_columns_list = list(common_columns)
    final_df = final_df[common_columns_list]

    for column in all_columns - common_columns:
        for df, prefix in zip(dfs, prefixes):
            if column in df.columns:
                # add this column to the final dataframe with prefix
                final_df[f"{prefix}_{column}"] = df[column]
    return final_df
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical_directories", nargs="+", help="Directories containing the blimp files")
    parser.add_argument("--historical_identifiers", nargs="+", help="Identifiers for the blimp files")
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument("--match_on", dest="match_on", default="pair_id")
    args = parser.parse_args()

    # get base directory
    args.output_directory = "/".join(args.output_directory.split("/")[:-1])

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    args.historical_results = ["/".join(result.split("/")[:-1]) for result in args.historical_directories]
    data = {}
    directory_identifiers = []
    for directory, identifier in zip(args.historical_results, args.historical_identifiers):
        directory_id = identifier
        files = glob.glob(f"{directory}/*.jsonl") + glob.glob(f"{directory}/*.json")
        for file in tqdm(files):
            file_id = file.split("/")[-1].split(".")[0]
            if file_id not in data:
                data[file_id] = {}
            if file.split(".")[-1] == "json":
                data[file_id][directory_id] = extract_data(file)
            #else:
            #    data[file_id][directory_id] = extract_data_results(file)


    for test_description, test_data in tqdm(data.items()):
        dfs = []
        prefixes = []
        for model_description, model_data in test_data.items():
            model_df = pd.DataFrame(model_data)
            dfs.append(model_df)
            prefixes.append(model_description)
        final_df = merge_dataframes(dfs, prefixes)
        final_file = f"{args.output_directory}/{test_description}.csv"
        final_df.to_csv(final_file, index=False)
