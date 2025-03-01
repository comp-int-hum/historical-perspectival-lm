import argparse
import subprocess
import os
import json
import pandas as pd
from tqdm import tqdm
import glob

from transformers import (
    GPT2Config, GPT2LMHeadModel, 
    LlamaConfig, LlamaForCausalLM, 
    GPTJConfig, GPTJForCausalLM,
    AutoModelForCausalLM, AutoTokenizer,
)
from transformers import AutoTokenizer
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
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt


def chunk_reader(fname, chunk_len=10000000):
    with open(fname, "rt") as f_in:
        while True:
            chunk = f_in.read(chunk_len)
            if not chunk:
                break
            yield chunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=str, help="models")
    parser.add_argument("--tokenizers", nargs="+", type=str, help="corresponding tokenizers")
    parser.add_argument("--time_data", nargs="+", type=str, help="time description")
    parser.add_argument("--test_sets", nargs="+", type=str, help="train_sets")
    parser.add_argument("--output", type=str, default="results.json", help="output file")
    args = parser.parse_args()

    # TODO fix this
    SEQUENCE_LENGTH = 128

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        num_train_epochs=1,
        eval_accumulation_steps=32,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        save_total_limit=1,  # Set to zero to avoid saving
        warmup_steps=300, 
        lr_scheduler_type="cosine",
        learning_rate=0.5,
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile = False,
    )
    output_directory = os.path.dirname(args.output)

    results = {}
    for model_file, tokenizer, model_time in zip(args.models, args.tokenizers, args.time_data):
        model = AutoModelForCausalLM.from_pretrained(model_file, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

        results[model_time] = {}
        for test_set, test_time in zip(args.test_sets, args.time_data):
            encoded = []
            for chunk in chunk_reader(test_set):
                enc_chunk = tokenizer.encode(chunk)
                encoded.extend(enc_chunk)
            test_dataset = GBDataset(torch.tensor(encoded), SEQUENCE_LENGTH, random_chunk=True)
            
            if len(test_dataset) > 10000:
                test_dataset = Subset(test_dataset, sample(range(len(test_dataset)), 10000))

            model.eval()
            model_trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                eval_dataset=test_dataset,
            )

            res_1 = model_trainer.evaluate()

            results[model_time][test_time] = res_1
            del model_trainer
        del model
        del tokenizer
        torch.cuda.empty_cache()

    
    cleaned_model = {}
    for model_time in results:
        cleaned_model[model_time] = []
        for test_time in results[model_time]:
            start_time = test_time.split("_")[0]
            end_time = test_time.split("_")[1]
            loss = results[model_time][test_time]["eval_loss"]
            perplexity = math.exp(loss)
            midpoint = (int(start_time) + int(end_time)) / 2
            cleaned_model[model_time].extend([(perplexity, midpoint)])

    

    for model_time in cleaned_model:
        x = [i[1] for i in cleaned_model[model_time]]
        y = [i[0] for i in cleaned_model[model_time]]
        plt.plot(x, y, label=f"model {model_time}")
    plt.ylabel("Perplexity")
    plt.xlabel("Time")
    plt.legend()
    plt.savefig(args.output.replace(".json", ".png"))

    with open(args.output, "w") as f:
        json.dump(results, f)
    