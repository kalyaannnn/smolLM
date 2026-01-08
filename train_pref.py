#!/usr/bin/env python3
"""
SmolLM DPO/APO (Direct/Anchored Preference Optimization) Script

Trains model on preference data (chosen vs rejected responses).

Usage:
    python train_pref.py --config configs/dpo.yaml \
        --base-model /path/to/sft_final.pt \
        --method dpo  # or --method apo
"""
import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import copy

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
    """Load DPO configuration."""
    default_config = {
        "model": {"base_checkpoint": None},
        "tokenizer": {"name": "meta-llama/Llama-2-7b-hf"},
        "data": {
            "dataset": "argilla/dpo-mix-7k",
            "split": "train",
            "max_samples": None,
        },
        "training": {
            "epochs": 1,
            "micro_batch_size": 8,
            "gradient_accumulation": 4,
            "seq_len": 2048,
            "precision": "bf16",
        },
        "dpo": {
            "beta": 0.1,
            "reference_free": False,
            "label_smoothing": 0.0,
            "loss_type": "sigmoid",
        },
        "optimizer": {
            "lr": 5e-7,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "betas": [0.9, 0.95],
        },
        "scheduler": {
            "warmup_ratio": 0.1,
            "min_lr_ratio": 0.0,
        },
        "logging": {
            "log_every_steps": 5,
            "project": "smol-lm",
            "use_wandb": True,
        },
        "checkpoint": {
            "save_every_steps": 100,
            "checkpoint_dir": "/content/drive/MyDrive/smol-lm-checkpoints/dpo",
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


def load_preference_dataset(tokenizer, config: Dict[str, Any]):
    """Load and prepare preference dataset."""
    from datasets import load_dataset
    
    data_config = config["data"]
    train_config = config["training"]
    
    dataset = load_dataset(
        data_config["dataset"],
        split=data_config["split"],
    )
    
    if data_config.get("max_samples"):
        dataset = dataset.select(range(min(len(dataset), data_config["max_samples"])))
    
    def tokenize_pair(examples):
        # Handle different dataset formats
        if "chosen" in examples and "rejected" in examples:
            chosen_texts = examples["chosen"]
            rejected_texts = examples["rejected"]
        elif "chosen_response" in examples:
            prompts = examples.get("prompt", [""] * len(examples["chosen_response"]))
            chosen_texts = [p + c for p, c in zip(prompts, examples["chosen_response"])]
            rejected_texts = [p + r for p, r in zip(prompts, examples["rejected_response"])]
        else:
            raise ValueError(f"Unknown dataset format. Keys: {list(examples.keys())}")
        
        # Handle if texts are lists (chat format)
        if isinstance(chosen_texts[0], list):
            chosen_texts = [" ".join([m.get("content", str(m)) for m in msgs]) for msgs in chosen_texts]
            rejected_texts = [" ".join([m.get("content", str(m)) for m in msgs]) for msgs in rejected_texts]
        
        chosen_tokens = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=train_config["seq_len"],
            padding="max_length",
            return_tensors="pt",
        )
        
        rejected_tokens = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=train_config["seq_len"],
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
    tokenized = dataset.map(
        tokenize_pair,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format("torch")
    
    return tokenized


def compute_log_probs(model, input_ids, attention_mask, labels):
    """Compute log probabilities for a sequence."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask.unsqueeze(1).unsqueeze(2),
    )
    logits = outputs["logits"]
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask padding
    mask = (shift_labels != -100) & (shift_labels != model.config.vocab_size)
    token_log_probs = token_log_probs * mask
    
    # Sum log probs
    return token_log_probs.sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
) -> tuple:
    """Compute DPO loss."""
    # Compute log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) \
                 - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "hinge":
        losses = F.relu(1 - beta * logits)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Compute metrics
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_margins = chosen_rewards - rejected_rewards
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    
    return losses.mean(), {
        "reward_margin": reward_margins.mean().item(),
        "reward_accuracy": reward_accuracies.mean().item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }


def train_dpo(config: Dict[str, Any], base_model_path: str, method: str = "dpo"):
    """Main DPO training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = config["training"]
    opt_config = config["optimizer"]
    dpo_config = config["dpo"]
    
    run_name = f"smol_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    print(f"Loading model from {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    if "config" in checkpoint:
        model_config = SmolLMConfig.from_dict(
            checkpoint["config"].get("model", checkpoint["config"])
        )
    else:
        model_config = SmolLMConfig.smol_600m()
    
    model_config.use_flash_attn = False  # Simpler for DPO
    
    # Create policy model
    policy_model = SmolLM(model_config)
    if "model_state" in checkpoint:
        policy_model.load_state_dict(checkpoint["model_state"])
    else:
        policy_model.load_state_dict(checkpoint)
    
    policy_model = policy_model.to(device)
    if train_config["precision"] == "bf16":
        policy_model = policy_model.bfloat16()
    
    # Create reference model (frozen copy)
    if not dpo_config["reference_free"]:
        print("Creating reference model...")
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
    else:
        reference_model = None
    
    # Load dataset
    print("Loading preference dataset...")
    dataset = load_preference_dataset(tokenizer, config)
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_config["micro_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
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
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print(f"\nStarting {method.upper()} training")
    print(f"Total steps: {total_steps}")
    print(f"Beta: {dpo_config['beta']}")
    
    policy_model.train()
    step = 0
    accumulation_count = 0
    metric_tracker = MetricTracker()
    
    dtype = torch.bfloat16 if train_config["precision"] == "bf16" else torch.float16
    
    for epoch in range(train_config["epochs"]):
        for batch in dataloader:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)
            
            with autocast(dtype=dtype):
                # Compute policy log probs
                policy_chosen_logps = compute_log_probs(
                    policy_model, chosen_ids, chosen_mask, chosen_ids
                )
                policy_rejected_logps = compute_log_probs(
                    policy_model, rejected_ids, rejected_mask, rejected_ids
                )
                
                # Compute reference log probs
                if reference_model is not None:
                    with torch.no_grad():
                        ref_chosen_logps = compute_log_probs(
                            reference_model, chosen_ids, chosen_mask, chosen_ids
                        )
                        ref_rejected_logps = compute_log_probs(
                            reference_model, rejected_ids, rejected_mask, rejected_ids
                        )
                else:
                    ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                
                # Compute loss
                loss, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=dpo_config["beta"],
                    label_smoothing=dpo_config["label_smoothing"],
                    loss_type=dpo_config["loss_type"],
                )
                loss = loss / train_config["gradient_accumulation"]
            
            loss.backward()
            accumulation_count += 1
            
            if accumulation_count % train_config["gradient_accumulation"] != 0:
                continue
            
            # Optimizer step
            grad_norm = clip_gradient_norm(policy_model, opt_config["grad_clip"])
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
                    "train/reward_margin": metrics["reward_margin"],
                    "train/reward_accuracy": metrics["reward_accuracy"],
                    "train/chosen_reward": metrics["chosen_reward"],
                    "train/rejected_reward": metrics["rejected_reward"],
                }
                logger.log(log_metrics, step=step)
                print(f"Step {step}/{total_steps} | Loss: {actual_loss:.4f} | "
                      f"Acc: {metrics['reward_accuracy']:.2%} | "
                      f"Margin: {metrics['reward_margin']:.3f}")
            
            # Checkpointing
            if checkpoint_manager.should_save(step):
                checkpoint_manager.save(
                    step=step,
                    model=policy_model,
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
    checkpoint_manager.save_final(policy_model, config, stage=method)
    logger.finish()
    print(f"{method.upper()} training complete!")


def main():
    parser = argparse.ArgumentParser(description="SmolLM DPO/APO Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "apo"])
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_dpo(config, args.base_model, args.method)


if __name__ == "__main__":
    main()


