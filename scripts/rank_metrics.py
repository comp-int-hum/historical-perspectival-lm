import argparse
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default = "Hplm/minimal_pairs", type=str, nargs="+", help="data to filter with")
    parser.add_argument("--model", type=str, help="model to use")
    parser.add_argument("--evaluation_result", type=str, help="Directory containing the blimp result files")
    parser.add_argument("--min_occurrence", type=int, default=2, help="How often a word must occur to be included")
    parser.add_argument("--output", type=str, help="Filtered results")
    args = parser.parse_args()
    
    # split the evaluation dataset based on start year and end year into ranges [(1750, 1820), (1820, 1850), (1850, 1880), (1880, 1910), (1910, 1940)]
    
    with open(args.data, "r") as df:
        dataset = datasets.load_dataset("Hplm/minimal_pairs", split = "test")
        


