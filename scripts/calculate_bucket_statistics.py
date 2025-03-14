import json
import argparse
import statistics
from collections import defaultdict

def compute_bucket_statistics(bucketed_data):
    """
    Computes mean and standard deviation of perplexity per model for each time bucket.
    Assumes that the same models are present in every example.
    """
    bucket_stats = {}

    for bucket, examples in bucketed_data.items():
        if not examples:
            bucket_stats[bucket] = {}
            continue

        expected_models = set(examples[0]["perplexities"].keys())

        for example in examples:
            assert set(example["perplexities"].keys()) == expected_models, \
                f"Inconsistent models found in bucket {bucket}. Expected {expected_models}, but got {set(example['perplexities'].keys())}"

        model_perplexities = defaultdict(list)

        for example in examples:
            for model, perplexity in example["perplexities"].items():
                model_perplexities[model].append(perplexity)

        bucket_stats[bucket] = {
            model: {
                "mean": statistics.mean(perplexities),
                "stddev": statistics.stdev(perplexities) if len(perplexities) > 1 else 0  # stddev is 0 if only one value
            }
            for model, perplexities in model_perplexities.items()
        }

    return bucket_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucketed_data", type=str, required=True, help="Path to the bucketed JSON file")
    parser.add_argument("--output_file", type=str, default="bucketed_stats.json", help="Output file for statistics")

    args = parser.parse_args()

    with open(args.bucketed_data, "r") as f:
        bucketed_data = json.load(f)

    bucket_stats = compute_bucket_statistics(bucketed_data)

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(bucket_stats, f)
