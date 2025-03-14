import argparse
import os
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data to filter with")
    parser.add_argument("--output", type=str, help="Filtered examples where obsolete examples are filtered out")
    args = parser.parse_args()
    
    # split the evaluation dataset based on start year and end year into ranges [(1750, 1820), (1820, 1850), (1850, 1880), (1880, 1910), (1910, 1940)]
    
    with open(args.data, "r") as df, open(args.output, "w") as of:
        examples = json.load(df)
        
        filtered_examples = [example for example in examples if example["doc"]["sense_end_year"] is None]
        
        json.dump(filtered_examples, of)
    
        