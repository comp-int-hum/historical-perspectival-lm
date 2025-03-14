import json
import argparse

def bucket_by_time(combined_data, time_boundaries):
    """
    Groups examples into time buckets based on their 'sense_start_year' using dynamic boundaries.
    """
    time_boundaries.sort()
    time_buckets = {}

    previous = None
    for boundary in time_boundaries:
        if previous is None:
            label = f"<{boundary}"
        else:
            label = f"{previous}-{boundary}"
        time_buckets[label] = []
        previous = boundary

    time_buckets[f"{previous}+"] = []

    for example in combined_data:
        start_year = example["doc"]["sense_start_year"]
        placed = False

        for boundary in time_boundaries:
            if start_year < boundary:
                bucket_label = f"<{boundary}" if boundary == time_boundaries[0] else f"{previous}-{boundary}"
                time_buckets[bucket_label].append(example)
                placed = True
                break
            previous = boundary

        if not placed:
            time_buckets[f"{previous}+"].append(example)  # Place in the final bucket

    return time_buckets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to the combined JSON data file")
    parser.add_argument("--time_boundaries", type=int, nargs="+", default=[1750, 1820, 1850, 1880, 1910, 1940], help="Time boundaries for the time slices")
    parser.add_argument("--output_file", type=str, help="Output file name")

    args = parser.parse_args()

    with open(args.data, "r") as f:
        combined_data = json.load(f)

    bucketed_data = bucket_by_time(combined_data, args.time_boundaries)

    with open(args.output_file, "w") as f:
        json.dump(bucketed_data, f)
