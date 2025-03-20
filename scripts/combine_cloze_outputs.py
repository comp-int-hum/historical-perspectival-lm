import json
import argparse

def load_and_combine_data(data_files, model_names, doc_filters, args_to_accumulate):
    combined_data = []

    task_outputs_by_model = {}
    for data_file, model_name in zip(data_files, model_names):
        with open(data_file, "r") as df:
            examples = json.load(df)    
            if doc_filters == "sense_end_year":
                filtered_examples = [ex for ex in examples if ex["doc"]["sense_end_year"] is None]
            else:
                filtered_examples = examples
            task_outputs_by_model[model_name] = filtered_examples

    assert all(len(ex) == len(task_outputs_by_model[model_names[0]]) for ex in task_outputs_by_model.values()), "Task output files have mismatched example counts"

    # Iterate over task outputs (assuming order is identical across files)
    for i, base_example in enumerate(task_outputs_by_model[model_names[0]]):

        accumulated_args = {
            arg: {model_name: task_outputs_by_model[model_name][i][arg] for model_name in model_names}
            for arg in args_to_accumulate
        }
        combined_entry = {
            "doc_id": base_example["doc_id"],
            "doc": base_example["doc"],
            "target": base_example["target"],
            "arguments": base_example["arguments"],
            "text": base_example["doc"]["text"],
            **accumulated_args
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
    parser.add_argument("--doc_filter", nargs="*", default=["sense_end_year"], help="List of document fields to filter out")
    parser.add_argument("--args_to_accumulate", nargs="+", default=["resps", "filtered_resps", "perplexity", "acc", "reciprocal_rank", "top_1", "top_5", "top_10", "top_20", "top_50", "top_100"], help="List of arguments to accumulate")
    args = parser.parse_args()
    
    assert len(args.data) == len(args.model_names), "Mismatch between task output files and model names"
    
    assert all(fp.endswith(".json") for fp in args.data)
    
    true_data_paths = [path[:-5] + ".jsonl" if path.endswith(".json") else path for path in args.data]
    
    # for path in true_data_paths:
    #     print(path)
    #     print()
        
    combined_data = load_and_combine_data(true_data_paths, args.model_names, args.doc_filter, args.args_to_accumulate)
    
    with open(args.output_file, "w") as outfile:
        json.dump(combined_data, outfile)