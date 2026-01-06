#!/usr/bin/env python3
"""
SmolLM SFT (Supervised Fine-Tuning) Script

Fine-tunes a pretrained model on instruction-following data.

Usage:
    python train_sft.py --config configs/sft.yaml \
        --base-model /path/to/pretrain_final.pt
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent))

from models.config import SmolLMConfig
from models.transformer import SmolLM
from data.tokenizer import load_tokenizer
from utils.logging import Logger, MetricTracker
from utils.checkpoint import CheckpointManager
from train.stability import StabilityMonitor, clip_gradient_norm


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load SFT configuration."""
    default_config = {
        "model": {"base_checkpoint": None},
        "tokenizer": {"name": "meta-llama/Llama-2-7b-hf"},
        "data": {
            "dataset": "HuggingFaceTB/smoltalk",
            "split": "train",
            "max_samples": None,
        },
        "training": {
            "epochs": 2,
            "micro_batch_size": 16,
            "gradient_accumulation": 4,
            "seq_len": 2048,
            "precision": "bf16",
        },
        "optimizer": {
            "lr": 2e-5,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "betas": [0.9, 0.95],
        },
        "scheduler": {
            "warmup_ratio": 0.03,
            "min_lr_ratio": 0.1,
        },
        "validation": {
            "eval_every_steps": 50,
            "eval_samples": 500,
        },
        "logging": {
            "log_every_steps": 10,
            "project": "smol-lm",
            "use_wandb": True,
        },
        "checkpoint": {
            "save_every_steps": 200,
            "checkpoint_dir": "/content/drive/MyDrive/smol-lm-checkpoints/sft",
            "keep_last_n": 2,
        },
    }
    
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        for section, values in user_config.items():
            if section in default_config and isinstance(values, dict):
                default_config[section].update(values)
            else:
                default_config[section] = values
    
    return default_config


def load_sft_dataset(tokenizer, config: Dict[str, Any]):
    """Load and prepare SFT dataset."""
    from datasets import load_dataset
    
    data_config = config["data"]
    train_config = config["training"]
    
    # Load dataset
    dataset = load_dataset(
        data_config["dataset"],
        split=data_config["split"],
        trust_remote_code=True,
    )
    
    if data_config.get("max_samples"):
        dataset = dataset.select(range(min(len(dataset), data_config["max_samples"])))
    
    # Tokenize
    def tokenize_function(examples):
        # Handle different dataset formats
        if "text" in examples:
            texts = examples["text"]
        elif "messages" in examples:
            # Chat format
            texts = []
            for messages in examples["messages"]:
                text = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text += f"<|{role}|>\n{content}\n"
                texts.append(text)
        elif "instruction" in examples and "response" in examples:
            texts = [
                f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
                for inst, resp in zip(examples["instruction"], examples["response"])
            ]
        else:
            # Fallback
            texts = [str(ex) for ex in examples[list(examples.keys())[0]]]
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=train_config["seq_len"],
            padding="max_length",
            return_tensors="pt",
        )
        
        tokenized["labels"] = tokenized["input_ids"].clone()
        # Mask padding tokens in labels
        tokenized["labels"][tokenized["attention_mask"] == 0] = -100
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset


def train_sft(config: Dict[str, Any], base_model_path: str):
    """Main SFT training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = config["training"]
    opt_config = config["optimizer"]
    
    run_name = f"smol_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup logging
    logger = Logger(
        project=config["logging"]["project"],
        run_name=run_name,
        config=config,
        use_wandb=config["logging"]["use_wandb"],
    )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config["checkpoint"]["checkpoint_dir"],
        save_every_steps=config["checkpoint"]["save_every_steps"],
        keep_last_n=config["checkpoint"]["keep_last_n"],
        run_name=run_name,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config["tokenizer"]["name"])
    
    # Load base model
    print(f"Loading base model from {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    if "config" in checkpoint:
        model_config = SmolLMConfig.from_dict(
            checkpoint["config"].get("model", checkpoint["config"])
        )
    else:
        model_config = SmolLMConfig.smol_600m()
    
    model_config.use_flash_attn = True
    model = SmolLM(model_config)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    if train_config["precision"] == "bf16":
        model = model.bfloat16()
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_sft_dataset(tokenizer, config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["micro_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    # Setup optimizer (just AdamW for SFT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config["lr"],
        weight_decay=opt_config["weight_decay"],
        betas=tuple(opt_config["betas"]),
    )
    
    # Setup scheduler
    total_steps = (
        len(dataloader) * train_config["epochs"] 
        // train_config["gradient_accumulation"]
    )
    warmup_steps = int(total_steps * config["scheduler"]["warmup_ratio"])
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return config["scheduler"]["min_lr_ratio"] + \
               (1 - config["scheduler"]["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
    
    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup scaler
    scaler = GradScaler() if train_config["precision"] == "fp16" else None
    
    # Training
    print(f"\nStarting SFT training")
    print(f"Total steps: {total_steps}")
    print(f"Epochs: {train_config['epochs']}")
    
    model.train()
    step = 0
    accumulation_count = 0
    
    stability_monitor = StabilityMonitor()
    metric_tracker = MetricTracker()
    
    for epoch in range(train_config["epochs"]):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            dtype = torch.bfloat16 if train_config["precision"] == "bf16" else torch.float16
            
            with autocast(dtype=dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
                    labels=labels,
                )
                loss = outputs["loss"] / train_config["gradient_accumulation"]
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_count += 1
            
            if accumulation_count % train_config["gradient_accumulation"] != 0:
                continue
            
            # Optimizer step
            if scaler:
                scaler.unscale_(optimizer)
            
            grad_norm = clip_gradient_norm(model, opt_config["grad_clip"])
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            actual_loss = loss.item() * train_config["gradient_accumulation"]
            metric_tracker.update("loss", actual_loss)
            
            if step % config["logging"]["log_every_steps"] == 0:
                log_metrics = {
                    "train/loss": actual_loss,
                    "train/grad_norm": grad_norm,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }
                logger.log(log_metrics, step=step)
                print(f"Step {step}/{total_steps} | Loss: {actual_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Checkpointing
            if checkpoint_manager.should_save(step):
                checkpoint_manager.save(
                    step=step,
                    model=model,
                    optimizers={"main": optimizer},
                    scheduler=scheduler,
                    config=config,
                )
            
            step += 1
            
            if step >= total_steps:
                break
        
        if step >= total_steps:
            break
    
    # Save final
    checkpoint_manager.save_final(model, config, stage="sft")
    logger.finish()
    print("SFT training complete!")


def main():
    parser = argparse.ArgumentParser(description="SmolLM SFT Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base-model", type=str, required=True, help="Path to pretrained checkpoint")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_sft(config, args.base_model)


if __name__ == "__main__":
    main()



