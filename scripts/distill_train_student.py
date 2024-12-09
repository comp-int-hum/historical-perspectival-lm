from transformers import (
    GPT2TokenizerFast,
    LlamaForCausalLM,
    LlamaConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Subset
from random import sample
import yaml

from pathlib import Path
import wandb
import argparse
from gb_dataloader import GBDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Path to the training data")
    parser.add_argument("--eval_data", help="Path to the evaluation data")
    parser.add_argument("--tokenizer_path", help="Path to the tokenizer")
    # teacher models
    parser.add_argument("--teacher_dir_1", help="Path to the first teacher model")
    parser.add_argument("--teacher_dir_2", help="Path to the second teacher model")
    # model parameters
    parser.add_argument("--config", type=str, default="./config/llama-16M.yaml", help="Configuration file path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    # wandb arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # output
    parser.add_argument("--output_dir", help="Path to the output directory")
    
    args, rest = parser.parse_known_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    tokenizer_path = args.tokenizer_path
    tokenizer = GPT2TokenizerFast(tokenizer_file= str(tokenizer_path))
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']

    if args.lr:
        config['training']['lr'] = args.lr

    # Dynamic Model Configuration
    if config['model']['type'] == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=2*tokenizer.model_max_length,
            hidden_size=config['model']['hidden_size'],
            intermediate_size=config['model']['intermediate_size'],
            num_hidden_layers=config['model']['n_layer'],
            num_attention_heads=config['model']['n_head'],
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        student = LlamaForCausalLM(model_config)
    else:
        raise ValueError(f"Model type {config['model']['type']} not supported for student model")

    # in the original code I had random_chunk = False
    # random_chunk=True is expected to improve the model performance a bit
    train_dataset = GBDataset(args.train_data, config['data']['seq_length'], random_chunk=True)
    full_eval_dataset = GBDataset(args.eval_data, config['data']['seq_length'], offset=0)

    eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
    eval_dataset = Subset(full_eval_dataset, eval_indices)


    teacher1 = LlamaForCausalLM.from_pretrained(args.teacher_dir_1)
    teacher2 = GPT2LMHeadModel.from_pretrained(args.teacher_dir_2)
    teachers = [teacher1, teacher2]


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )


    print(f'model num parameters: student = {student.num_parameters()}')
    print(f'model num parameters: teacher1 = {teacher1.num_parameters()}')
    print(f'model num parameters: teacher2 = {teacher2.num_parameters()}')



    #  Distillation Trainer
    #  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
    # to work with an ensemble of teachers


    class DistillationTrainingArguments(TrainingArguments):
        def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha = alpha
            self.temperature = temperature


    class DistillationTrainer(Trainer):
        def __init__(self, *args, teacher_models=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.teachers = teacher_models
            for teacher in self.teachers:
                # place each teacher on same device as student
                self._move_model_to_device(teacher, self.model.device)
                teacher.eval()

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=0):
            # compute student output
            outputs_student = model(**inputs)
            student_loss = outputs_student.loss

            # compute teacher output
            with torch.no_grad():
                all_teacher_logits = []
                for teacher in self.teachers:
                    outputs_teacher = teacher(**inputs)
                    all_teacher_logits.append(outputs_teacher.logits)
                avg_teacher_logits = torch.stack(all_teacher_logits).mean(dim=0)

            # assert size
            assert outputs_student.logits.size() == avg_teacher_logits.size()

            # Soften probabilities and compute distillation loss
            loss_function = nn.KLDivLoss(reduction="batchmean")
            loss_logits = (
                loss_function(
                    F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                    F.softmax(avg_teacher_logits / self.args.temperature, dim=-1),
                )
                * (self.args.temperature ** 2)
            )
            # Return weighted student loss
            loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
            return (loss, outputs_student) if return_outputs else loss


    if args.use_wandb:
        wandb.login()
        wandb.init(project=args.wandb_project, name=args.wandb_name)


    output_dir = args.output_dir
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    training_args = DistillationTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        num_train_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        save_total_limit=1,  # Set to zero to avoid saving
        report_to="wandb",
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        logging_steps=20,
        fp16=config['training']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=float(config['training']['weight_decay']),
        alpha=float(config['training']['alpha']),
        temperature=float(config['training']['temperature']),
    )


    trainer = DistillationTrainer(
            student,
            training_args,
            teacher_models=teachers,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,

        )


    trainer.train()


    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)