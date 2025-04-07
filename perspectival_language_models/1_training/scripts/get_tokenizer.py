import argparse
import logging
from transformers import AutoTokenizer, LlamaTokenizer
import transformers
import torch
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="text input")
    parser.add_argument("--output", help="Tokenizer model output")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    tokenizer = AutoTokenizer.from_pretrained(args.input, use_fast = True)
    tokenizer.save_pretrained(args.output)

    





