import argparse
import logging
from transformers import AutoTokenizer, LlamaTokenizer
import transformers
import torch
import os
# code restructured from: https://github.com/timinar/BabyLlama/blob/main/cleaning_and_tokenization.ipynb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="text input")
    parser.add_argument("--output", help="Tokenizer model output")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    
    # print(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.input, use_fast = True)
    # args.output = os.path.dirname(args.output)
    # print(f"OUTPUT: {args.output}")
    tokenizer.save_pretrained(args.output)

    # model_id = "meta-llama/Meta-Llama-3-8B"

    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto",
    # )

    # pipeline.tokenizer.save_pretrained(args.output)

    





