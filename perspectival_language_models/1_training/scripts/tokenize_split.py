import argparse
from tokenizers import Tokenizer
from transformers import AutoTokenizer
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for chunk in chunk_reader(args.input):
        enc_chunk = tokenizer.encode(chunk)
        encoded.extend(enc_chunk)
        
    logging.info(f"{len(encoded)} tokens saved to {args.output}")
    torch.save(torch.tensor(encoded), args.output)

    




