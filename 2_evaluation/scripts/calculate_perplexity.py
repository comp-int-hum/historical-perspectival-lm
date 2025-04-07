import argparse
import os
import json

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
)
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import argparse
from gb_dataloader import GBDataset
import torch
from collections import defaultdict


def chunk_reader(fname, chunk_len=10000000):
    with open(fname, "rt") as f_in:
        while True:
            chunk = f_in.read(chunk_len)
            if not chunk:
                break
            yield chunk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="models")
    parser.add_argument("--tokenizer", type=str, help="corresponding tokenizers")
    parser.add_argument("--split", type=str, help="split description")
    parser.add_argument("--test_sets", nargs="+", type=str, help="test_set")
    parser.add_argument("--test_splits", nargs="+", type=str, help="test_split")
    parser.add_argument("--output", type=str, help="output file")
    args = parser.parse_args()


    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        save_strategy = "epoch",
        eval_strategy = "no",
        num_train_epochs=1,
        eval_accumulation_steps=32,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        save_total_limit=1,
        warmup_steps=300, 
        lr_scheduler_type="cosine",
        learning_rate=0.5,
        logging_steps=20,
        fp16=True,
        load_best_model_at_end=False,
        torch_compile = False,
        report_to=[],
    )
    output_directory = os.path.dirname(args.output)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
    model.eval()
    model_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator
    )

    SEQUENCE_LENGTH = tokenizer.model_max_length

    results = defaultdict(dict)

    for test_set, test_split in zip(args.test_sets, args.test_splits):
        encoded = []
        for chunk in chunk_reader(test_set):
            enc_chunk = tokenizer.encode(chunk)
            encoded.extend(enc_chunk)
        test_dataset = GBDataset(torch.tensor(encoded), SEQUENCE_LENGTH, random_chunk=True)
        evaluation_result = model_trainer.evaluate(eval_dataset = test_dataset)
        results[args.split][test_split] = evaluation_result
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    