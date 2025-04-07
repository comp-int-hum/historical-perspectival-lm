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
            "sentence_good": doc.get("sentence_good"),
            "sentence_bad": doc.get("sentence_bad"),
            #"field": doc.get("field"),
            #"linguistics_term": doc.get("linguistics_term"),
            #"UID": doc.get("UID"),
            #"simple_LM_method": doc.get("simple_LM_method"),
            #"one_prefix_method": doc.get("one_prefix_method"),
            #"two_prefix_method": doc.get("two_prefix_method"),
            #"lexically_identical": doc.get("lexically_identical"),
            "pair_id": doc.get("pairID") if "pairID" in doc else doc.get("pair_id"),
            #"target": entry.get("target"),
            "acc": entry.get("acc"),
            # Store nested structures as strings if needed
            #"arguments": json.dumps(entry.get("arguments", [])),
            "perplexity_good": response[0][0][0],
            "perplexity_bad": response[1][0][0],
            "response_good": response[0][0][1],
            "response_bad": response[1][0][1],
        }
        flattened_rows.append(row)
    return flattened_rows

# extract data from results file
def extract_data_results(file):
    with open(file, 'r') as f:
        data = json.load(f)["results"]
    flattened_rows = []
    for entry, values in data.items():
        row = {
            "acc" : values["acc,none"],
            "acc_stderr" : values["acc_stderr,none"],
            "alias" : values["alias"],
        }
        flattened_rows.append(row)
    return flattened_rows


def merge_dataframes(dfs, prefixes, index_to_merge_on="pair_id"):
    assert len(dfs) == len(prefixes), "Number of dataframes must match number of prefixes"
    all_columns = set()

    all_values_to_merge_on = {}
    for df in dfs:
        all_columns.update(df.columns)
        if index_to_merge_on in df:
            for i in df[index_to_merge_on].values:
                if i not in all_values_to_merge_on:
                    all_values_to_merge_on[i] = {"sentence_good": df["sentence_good"].values[df[index_to_merge_on].values == i][0],
                                                "sentence_bad": df["sentence_bad"].values[df[index_to_merge_on].values == i][0]}
                else:
                    assert all_values_to_merge_on[i]["sentence_good"] == df["sentence_good"].values[df[index_to_merge_on].values == i][0]
                    assert all_values_to_merge_on[i]["sentence_bad"] == df["sentence_bad"].values[df[index_to_merge_on].values == i][0]

    # fill in empty lines for missing values
    for i, df in enumerate(dfs):
        new_rows = []
        if index_to_merge_on in df:
            for value in all_values_to_merge_on.keys():
                if value not in df[index_to_merge_on].values:
                    new_row = {index_to_merge_on: value}
                    for column in all_columns:
                        if column != index_to_merge_on:
                            new_row[column] = None
                    new_row["sentence_good"] = all_values_to_merge_on[value]["sentence_good"]
                    new_row["sentence_bad"] = all_values_to_merge_on[value]["sentence_bad"]
                    new_rows.append(new_row)
            dfs[i] = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            assert set(dfs[i][index_to_merge_on].values) == set(all_values_to_merge_on.keys())
            dfs[i].sort_values(by=index_to_merge_on, inplace=True)
            dfs[i].reset_index(drop=True, inplace=True)

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
    parser.add_argument("--blimp_directories", nargs="+", help="Directories containing the blimp files")
    parser.add_argument("--blimp_identifiers", nargs="+", help="Identifiers for the blimp files")
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument("--match_on", dest="match_on", default="pair_id")
    args = parser.parse_args()

    # get base directory
    args.output_directory = "/".join(args.output_directory.split("/")[:-1])

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    args.blimp_results = ["/".join(result.split("/")[:-1]) for result in args.blimp_directories]
    data = {}
    directory_identifiers = []
    for directory, identifier in zip(args.blimp_results, args.blimp_identifiers):
        directory_id = identifier
        files = glob.glob(f"{directory}/*.jsonl") + glob.glob(f"{directory}/*.json")
        for file in tqdm(files):
            file_id = file.split("/")[-1].split(".")[0]
            if file_id not in data:
                data[file_id] = {}
            if file.split(".")[-1] == "jsonl":
                data[file_id][directory_id] = extract_data(file)
            else:
                data[file_id][directory_id] = extract_data_results(file)


    for test_description, test_data in tqdm(data.items()):
        dfs = []
        prefixes = []
        for model_description, model_data in test_data.items():
            model_df = pd.DataFrame(model_data)
            dfs.append(model_df)
            prefixes.append(model_description)
        final_df = merge_dataframes(dfs, prefixes, index_to_merge_on=args.match_on)
        final_file = f"{args.output_directory}/{test_description}.csv"
        final_df.to_csv(final_file, index=False)
