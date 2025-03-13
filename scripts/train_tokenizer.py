import argparse
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC
import logging
from transformers import GPT2TokenizerFast


# code restructured from: https://github.com/timinar/BabyLlama/blob/main/cleaning_and_tokenization.ipynb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="text input")
    parser.add_argument("--output", help="Tokenizer model output")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    logging.info("Training tokenizer")
    trainer = trainers.BpeTrainer(vocab_size=16000, min_frequency=2, special_tokens=["<pad>", "<s>", "</s>"])
    tokenizer.train([args.input], trainer)

    logging.info("Saving tokenizer model")
    tokenizer.save(args.output + ".json", pretty=True)

    gpt2_tokenizer = GPT2TokenizerFast(tokenizer_file=args.output + ".json")
    gpt2_tokenizer.save_pretrained(args.output)

    logging.info("Testing tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.output)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"


    # text = 'Shiro Okada (岡田志郎, "Okada Shirō", June 9, 1949; Hirakata, Osaka {age 71} - ) is a Japanese guitarist who participate in the Group Sound band, the Ox. His nickname was Shiro (シロー) and his real name is Shiro Okamoto (岡田史郎).'
    text = "The quick brown fox jumps over the lazy dog."

    encoded = tokenizer.encode(text)
    logging.info(f"Encoded String: {encoded}")

    decoded = tokenizer.decode(encoded)
    logging.info(f"Decoded String: {decoded}")



    





