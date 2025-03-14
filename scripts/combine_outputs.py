import json
import argparse

# def load_and_combine_data(data_files, model_names):
#     combined_data = []

#     examples_by_model = []
#     for data_file in data_files:
#         with open(data_file, "r") as df:
#             examples = json.load(df)
#             filtered_examples = []
#             for ex in examples:
#                 print(f"EXAMPLE IS HERE: {ex}")
#                 if ex["doc"]["sense_end_year"] is None:
#                     filtered_examples.append(ex)
#             # filtered_examples = [ex for ex in examples if ex["doc"]["sense_end_year"] is None]
#             examples_by_model.append(filtered_examples)

#     assert all(len(ex) == len(examples_by_model[0]) for ex in examples_by_model), "Data files have mismatched example counts"

#     # Iterate over examples (assuming order is identical across files)
#     for idx in range(len(examples_by_model[0])):
#         base_example = examples_by_model[0][idx]

#         # Collect perplexities from each model
#         perplexity_dict = {model_names[i]: examples_by_model[i][idx]["perplexity"] for i in range(len(model_names))}

#         # Create combined entry
#         combined_entry = {
#             "doc_id": base_example["doc_id"],
#             "text": base_example["doc"]["text"],
#             "word": base_example["doc"]["word"],
#             "target": base_example["target"],
#             "perplexities": perplexity_dict  # Stores perplexities for all models
#         }

#         combined_data.append(combined_entry)

#     return combined_data

def load_and_combine_data(data_files, model_names):
    combined_data = []

    task_outputs_by_model = {}
    for data_file, model_name in zip(data_files, model_names):
        with open(data_file, "r") as df:
            examples = json.load(df)    
            filtered_examples = [ex for ex in examples if ex["doc"]["sense_end_year"] is None]
            task_outputs_by_model[model_name] = filtered_examples

    assert all(len(ex) == len(task_outputs_by_model[model_names[0]]) for ex in task_outputs_by_model.values()), "Task output files have mismatched example counts"

    # Iterate over task outputs (assuming order is identical across files)
    for i, base_example in enumerate(task_outputs_by_model[model_names[0]]):

        perplexity_dict = {model_name: task_outputs_by_model[model_name][i]["perplexity"] for model_name in model_names}

        combined_entry = {
            "doc_id": base_example["doc_id"],
            "doc": base_example["doc"],
            "target": base_example["target"],
            "arguments": base_example["arguments"],
            "text": base_example["doc"]["text"],
            "perplexities": perplexity_dict
        }

        combined_data.append(combined_entry)

    return combined_data


def save_combined_data(combined_data, output_file):
    """
    Saves the combined data into a single JSON file.
    """
    with open(output_file, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="List of JSON data files")
    parser.add_argument("--model_names", nargs="+", required=True, help="List of model names corresponding to the data files")
    parser.add_argument("--output_file", type=str, default="combined_data.json", help="Output file name")
    args = parser.parse_args()
    
    assert len(args.data) == len(args.model_names), "Mismatch between task output files and model names"
    
    assert all(fp.endswith(".json") for fp in args.data)
    
    true_data_paths = [path[:-5] + ".jsonl" if path.endswith(".json") else path for path in args.data]
    
    # for path in true_data_paths:
    #     print(path)
    #     print()
        
    combined_data = load_and_combine_data(true_data_paths, args.model_names)
    
    with open(args.output_file, "w") as outfile:
        json.dump(combined_data, outfile)