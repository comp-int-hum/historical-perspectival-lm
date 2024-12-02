import argparse
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast
import torch
import logging



def chunk_reader(fname, chunk_len=10000000):
    with open(fname, "rt") as f_in:
        while True:
            chunk = f_in.read(chunk_len)
            if not chunk:
                break
            yield chunk
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="split text input")
    parser.add_argument("--tokenizer", help="pretrained tokenizer json")
    parser.add_argument("--output", help="tokenized file output")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    logging.info(f"Encoding file: {args.input}")

    encoded = []
    tokenizer = GPT2TokenizerFast(tokenizer_file=args.tokenizer)
    for chunk in chunk_reader(args.input):
        enc_chunk = tokenizer.encode(chunk)
        encoded.extend(enc_chunk)
        
    #with open(args.input, "rt") as in_split:
    #    encoded = tokenizer.encode(in_split.read())

    logging.info(f"{len(encoded)} tokens saved to {args.output}")
    torch.save(torch.tensor(encoded), args.output)

    




