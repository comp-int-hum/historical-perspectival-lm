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
import glob
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default=None, help="Path to the training data")
    parser.add_argument("--eval_data", type=str, default=None, help="Path to the evaluation data")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer")
    # model parameters
    parser.add_argument("--config", type=str, default="./config/llama-360M.yaml", help="Configuration file path")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    # wandb arguments
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # output
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed(args.random_seed)


    # Override config parameters if provided as command-line arguments
    if args.lr:
        config['training']['lr'] = args.lr


    train_dataset = GBDataset(args.train_data, config['data']['seq_length'], random_chunk=True)

    if config['training'].get('gpus', None) is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = config['training']['gpus']
    #TODO: change back
    print(f"using {config['training']['gpus']} GPUs")
    train_tokens = len(train_dataset) * config['data']['seq_length']
    print(f"train_tokens = {train_tokens/10**6}M")
    full_eval_dataset = GBDataset(args.eval_data, config['data']['seq_length'], offset=0)

    
    eval_indices = sample(range(len(full_eval_dataset)), config['data']['eval_samples'])
    eval_dataset = Subset(full_eval_dataset, eval_indices)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_path)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"
    tokenizer.model_max_length = config['data']['seq_length']

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Dynamic Model Configuration
    if config['model']['type'] == "Llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=2*tokenizer.model_max_length,
            hidden_size=config['model']['hidden_size'],
            intermediate_size=config['model']['intermediate_size'],
            num_hidden_layers=config['model']['n_layer'],
            num_attention_heads=config['model']['n_head'],
            num_key_value_heads=config['model'].get('n_KV', config['model']['n_head']),
            tie_word_embeddings=config['model'].get('tie_word_embeddings', False),
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
            attention_dropout=config['model'].get('attention_dropout', 0.0),
        )
        model = LlamaForCausalLM(model_config)
    elif config['model']['type'] == "GPT2":
        model_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=2*tokenizer.model_max_length,
            n_embd=config['model']['hidden_size'],
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            resid_pdrop = config['model']['resid_pdrop'],
            embd_pdrop = config['model']['embd_pdrop'],
            attn_pdrop = config['model']['attn_pdrop'],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPT2LMHeadModel(model_config)
    elif config['model']['type'] == "GPTJ":
        model_config = GPTJConfig(
            vocab_size=tokenizer.vocab_size,
            n_positions=2*tokenizer.model_max_length,
            n_embd=config['model']['hidden_size'],
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            resid_pdrop = config['model']['resid_pdrop'],
            embd_pdrop = config['model']['embd_pdrop'],
            attn_pdrop = config['model']['attn_pdrop'],
            tie_word_embeddings=config['model']['tie_word_embeddings'],
            pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        )
        model = GPTJForCausalLM(model_config)

    print(f'model parameters = {model.num_parameters()}')


    output_dir = args.output_dir
    accumulation_steps = config['training']['gradient_accumulation_steps']
    per_device_bsz = config['training']['batch_size'] // accumulation_steps

    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"training length: {len(train_dataset)}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        num_train_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=accumulation_steps,
        per_device_train_batch_size=per_device_bsz,
        per_device_eval_batch_size=per_device_bsz,
        save_total_limit=1,  # Set to zero to avoid saving
        warmup_steps=config['training']['warmup_steps'], 
        lr_scheduler_type="cosine",
        learning_rate=float(config['training']['lr']),
        logging_steps=20,
        fp16=config['training']['fp16'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        torch_compile = config['training'].get('torch_compile', False),
        save_only_model=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.use_wandb:
        wandb.login()
        wandb.init(project= args.wandb_project, name=args.wandb_name, config=config)

    trainer.train()

    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint, ignore_errors=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)