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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default=None, help="Path to the test data")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer")
    # model parameters
    parser.add_argument("--teacher_dir_1", type=str, default=None, help="Path to the first teacher model")
    parser.add_argument("--teacher_dir_2", type=str, default=None, help="Path to the second teacher model")
    parser.add_argument("--student_dir", type=str, default=None, help="Path to the student model")
    # output
    parser.add_argument("--report", type=str, default=None, help="Path to the report")
    args = parser.parse_args()
    print(args)

    teacher1 = LlamaForCausalLM.from_pretrained(args.teacher_dir_1)
    teacher2 = LlamaForCausalLM.from_pretrained(args.teacher_dir_2)
    student = LlamaForCausalLM.from_pretrained(args.student_dir)

    # TODO fix this
    SEQUENCE_LENGTH = 128

    test_dataset = GBDataset(args.test_data, SEQUENCE_LENGTH, random_chunk=True)

    test_token = len(test_dataset) * SEQUENCE_LENGTH
    print(f"train_tokens = {test_token/10**6}M")

    tokenizer = GPT2TokenizerFast(tokenizer_file = str(args.tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = SEQUENCE_LENGTH
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        num_train_epochs=1,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=32,
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

    teacher1_trainer = Trainer(
        model=teacher1,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )

    teacher2_trainer = Trainer(
        model=teacher2,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )

    student_trainer = Trainer(
        model=student,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )

    res_1 = teacher1_trainer.evaluate()
    res_2 = teacher2_trainer.evaluate()
    res_3 = student_trainer.evaluate()

    with open(args.report, "wt") as fout:
        fout.write(f"Teacher 1: {res_1['eval_loss']}")
        fout.write(f"Teacher 2: {res_2['eval_loss']}")
        fout.write(f"Student: {res_3['eval_loss']}")