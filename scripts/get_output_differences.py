import argparse
import json

def find_large_differences(combined_data, model1, model2, arg_name, threshold):
    """
    Finds all examples where the difference between two models for a given argument is greater than the threshold.
    Separates examples where model1 has a higher value than model2 and vice versa.

    Parameters:
    - combined_data: List of dictionaries containing accumulated data.
    - model1, model2: The two model names to compare.
    - arg_name: The argument name to compare.
    - threshold: The minimum absolute difference to consider as "very large".

    Returns:
    - model1_higher: List of examples where model1's value is greater than model2's.
    - model2_higher: List of examples where model2's value is greater than model1's.
    """
    model1_higher = []
    model2_higher = []

    for example in combined_data:
        if arg_name in example and model1 in example[arg_name] and model2 in example[arg_name]:
            value1 = example[arg_name][model1]
            value2 = example[arg_name][model2]
            difference = abs(value1 - value2)

            if difference > threshold:
                example_data = {
                    "doc_id": example["doc_id"],
                    "text": example["text"],
                    "target": example["target"],
                    "arguments": example["arguments"],
                    "model1_value": value1,
                    "model2_value": value2,
                    "difference": difference
                }

                if value1 > value2:
                    model1_higher.append(example_data)
                else:
                    model2_higher.append(example_data)

    return model1_higher, model2_higher


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help = "Path to the json file containing combined data.")
    parser.add_argument("model1", type=str, help = "First model name")
    parser.add_argument("model2", type=str, help = "Second model name")
    parser.add_argument("metric", type=str, help = "Metric to compare")
    parser.add_argument("threshold", type=float, help = "Threshold")
    parser.add_argument("--output_file", type=str, help = "Output file name")

    args = parser.parse_args()

    with open(args.data_file, "r") as f:
        bucketed_data = json.load(f)
        
    output_dict = {}
    for bucket, examples in bucketed_data.items():
        large_diff_examples = find_large_differences(examples, args.model1, args.model2, args.metric, args.threshold)
        output_dict[bucket] = large_diff_examples
        
    with open(args.output_file, "w") as f:
        json.dump(output_dict, f)
    