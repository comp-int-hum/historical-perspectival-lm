from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM
)
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2TokenizerFast
from torch.utils.data import Subset
from random import sample, seed
from pathlib import Path
import yaml
import argparse
import wandb
from gb_dataloader import GBDataset
import torch
import json
import os
import pandas as pd
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal_pairs", type=str, default=None, help="Path to the test data")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the tokenizer")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to the first teacher model")
    parser.add_argument("--report", type=str, default=None, help="Path to the report")
    args = parser.parse_args()
    print(args)

    minimal_pairs = pd.read_csv(args.minimal_pairs, sep=";")
    
    model = LlamaForCausalLM.from_pretrained(args.model_dir)
    tokenizer = GPT2TokenizerFast(tokenizer_file = str(args.tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)


    results = []
    for index, row in tqdm(minimal_pairs.iterrows()):

        sentence_1 = row["sent_1"]
        sentence_2 = row["sent_2"]
        
        inputs_1 = tokenizer(sentence_1, return_tensors="pt")
        inputs_2 = tokenizer(sentence_2, return_tensors="pt")

        inputs_1 = inputs_1.to(device)
        inputs_2 = inputs_2.to(device)

        outputs_1 = model(**inputs_1, labels=inputs_1["input_ids"])
        log_perplexity_1 = outputs_1.loss

        outputs_2 = model(**inputs_2, labels=inputs_2["input_ids"])
        log_perplexity_2 = outputs_2.loss

        # TODO: check if True/False difference is truly >= 0 
        
        result_dict = {
            "task": "minimal_pairs",
            "doc_id": index,
            "doc": {
                "sentence_one": sentence_1,
                "sentence_two": sentence_2,
                "stem": row["stem"],
                "date": row["date"],
                "rationale": row["rationale"]
            },
            "target": "NA",
            "arguments": [
                [
                    "", sentence_1
                ],
                [
                    "", sentence_2
                ]
            ],
            "resps": [
                [
                    [
                        log_perplexity_1.item(),
                        log_perplexity_1.item()>=0
                    ],
                ],
                [
                    [
                        log_perplexity_2.item(),
                        log_perplexity_2.item()>=0
                    ]
                ]
            ],
            "acc": "NA",
            "more_likely_sentence": (
                "sentence_1" if log_perplexity_1.item()>=log_perplexity_2.item()
                else "sentence_2"
            )
        }
        
        results.append(result_dict)
        
    with open(args.report, "wt") as fout:
        # add to jsonl
        json.dump(results, fout)