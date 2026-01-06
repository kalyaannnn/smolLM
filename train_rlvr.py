#!/usr/bin/env python3
"""
SmolLM RLVR (RL from Verifiable Rewards) Script

Optional training stage that uses verifiable rewards (e.g., math answer correctness,
code execution results) to further improve the model.

Usage:
    python train_rlvr.py --config configs/rlvr.yaml \
        --base-model /path/to/dpo_final.pt \
        --verifier-model /path/to/verifier

Note: This is a simplified implementation. For production RLVR,
consider using TRL's PPOTrainer or similar frameworks.
"""
import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.config import SmolLMConfig
from models.transformer import SmolLM
from data.tokenizer import load_tokenizer
from utils.logging import Logger
from utils.checkpoint import CheckpointManager


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load RLVR configuration."""
    default_config = {
        "model": {"base_checkpoint": None},
        "verifier": {"model_path": None},
        "tokenizer": {"name": "meta-llama/Llama-2-7b-hf"},
        "data": {
            "dataset": "openai/gsm8k",
            "split": "train",
            "max_samples": 5000,
        },
        "training": {
            "epochs": 1,
            "micro_batch_size": 4,
            "gradient_accumulation": 8,
            "seq_len": 2048,
            "precision": "bf16",
            "num_generations_per_prompt": 4,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "rlvr": {
            "ppo_epochs": 2,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.1,
            "init_kl_coef": 0.1,
            "target_kl": 6.0,
            "reward_baseline": "mean",
            "reward_scale": 1.0,
        },
        "optimizer": {
            "lr": 1e-6,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
        },
        "logging": {
            "log_every_steps": 5,
            "project": "smol-lm",
            "use_wandb": True,
        },
        "checkpoint": {
            "save_every_steps": 50,
            "checkpoint_dir": "/content/drive/MyDrive/smol-lm-checkpoints/rlvr",
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


class SimpleVerifier:
    """
    Simple verifier for math problems.
    
    For GSM8K: checks if the final answer matches.
    For code: would execute and check output.
    """
    
    def __init__(self, verifier_model_path: Optional[str] = None):
        self.verifier_model_path = verifier_model_path
        # Could load a classifier model here
    
    def verify_math(self, response: str, answer: str) -> float:
        """Verify math answer. Returns reward in [0, 1]."""
        # Extract final number from response
        import re
        
        # Look for "#### number" pattern (GSM8K format)
        match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', response)
        if match:
            predicted = match.group(1).replace(',', '')
        else:
            # Look for last number in response
            numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
            if numbers:
                predicted = numbers[-1].replace(',', '')
            else:
                return 0.0
        
        # Clean answer
        answer_clean = re.sub(r'[^\d.-]', '', str(answer))
        
        try:
            if float(predicted) == float(answer_clean):
                return 1.0
            else:
                return 0.0
        except ValueError:
            return 0.0
    
    def __call__(self, responses: List[str], answers: List[str]) -> List[float]:
        """Compute rewards for batch of responses."""
        return [self.verify_math(r, a) for r, a in zip(responses, answers)]


def train_rlvr(
    config: Dict[str, Any],
    base_model_path: str,
    verifier_model_path: Optional[str] = None,
):
    """
    Simplified RLVR training loop.
    
    For production use, consider TRL's PPOTrainer which handles:
    - Proper advantage estimation
    - KL penalty scheduling
    - Value function training
    - Experience replay
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = config["training"]
    rlvr_config = config["rlvr"]
    
    run_name = f"smol_rlvr_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    
    # Load model
    print(f"Loading model from {base_model_path}...")
    checkpoint = torch.load(base_model_path, map_location=device)
    
    if "config" in checkpoint:
        model_config = SmolLMConfig.from_dict(
            checkpoint["config"].get("model", checkpoint["config"])
        )
    else:
        model_config = SmolLMConfig.smol_600m()
    
    model_config.use_flash_attn = False
    model = SmolLM(model_config)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    if train_config["precision"] == "bf16":
        model = model.bfloat16()
    
    # Load verifier
    print("Initializing verifier...")
    verifier = SimpleVerifier(verifier_model_path)
    
    # Load dataset
    print("Loading dataset...")
    from datasets import load_dataset
    
    dataset = load_dataset(
        config["data"]["dataset"],
        "main",
        split=config["data"]["split"],
    )
    
    if config["data"].get("max_samples"):
        dataset = dataset.select(range(min(len(dataset), config["data"]["max_samples"])))
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    
    # Training loop
    print(f"\nStarting RLVR training")
    print(f"Dataset size: {len(dataset)}")
    print(f"Generations per prompt: {train_config['num_generations_per_prompt']}")
    
    step = 0
    total_rewards = []
    
    for epoch in range(train_config["epochs"]):
        for i, example in enumerate(dataset):
            # Get prompt and answer
            question = example["question"]
            answer = example["answer"]
            
            # Format prompt
            prompt = f"Question: {question}\n\nAnswer: Let me solve this step by step.\n"
            
            # Generate responses
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            responses = []
            log_probs_list = []
            
            model.eval()
            with torch.no_grad():
                for _ in range(train_config["num_generations_per_prompt"]):
                    output = model.generate(
                        input_ids,
                        max_new_tokens=train_config["max_new_tokens"],
                        temperature=train_config["temperature"],
                        top_p=train_config["top_p"],
                        do_sample=True,
                    )
                    
                    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
                    responses.append(response_text)
            
            # Compute rewards
            rewards = verifier(responses, [answer] * len(responses))
            avg_reward = sum(rewards) / len(rewards)
            total_rewards.append(avg_reward)
            
            # Simple REINFORCE update (for demonstration)
            # Production RLVR would use PPO with advantage estimation
            model.train()
            
            for response_text, reward in zip(responses, rewards):
                if reward > 0:  # Only learn from correct responses
                    response_ids = tokenizer.encode(
                        response_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=train_config["seq_len"],
                    ).to(device)
                    
                    outputs = model(response_ids, labels=response_ids)
                    loss = outputs["loss"] * (reward - 0.5)  # Baseline subtraction
                    loss.backward()
            
            # Optimizer step
            if (i + 1) % train_config["gradient_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["optimizer"]["grad_clip"],
                )
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                # Logging
                if step % config["logging"]["log_every_steps"] == 0:
                    recent_rewards = total_rewards[-100:] if len(total_rewards) >= 100 else total_rewards
                    avg_recent = sum(recent_rewards) / len(recent_rewards)
                    
                    log_metrics = {
                        "train/reward": avg_reward,
                        "train/avg_reward_100": avg_recent,
                        "train/num_correct": sum(1 for r in rewards if r > 0),
                    }
                    logger.log(log_metrics, step=step)
                    print(f"Step {step} | Reward: {avg_reward:.2f} | "
                          f"Avg(100): {avg_recent:.2f} | "
                          f"Correct: {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
                
                # Checkpointing
                if checkpoint_manager.should_save(step):
                    checkpoint_manager.save(
                        step=step,
                        model=model,
                        optimizers={"main": optimizer},
                        scheduler=None,
                        config=config,
                    )
    
    # Save final
    checkpoint_manager.save_final(model, config, stage="rlvr")
    logger.finish()
    
    final_avg = sum(total_rewards) / len(total_rewards) if total_rewards else 0
    print(f"\nRLVR training complete!")
    print(f"Final average reward: {final_avg:.3f}")


def main():
    parser = argparse.ArgumentParser(description="SmolLM RLVR Training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--verifier-model", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_rlvr(config, args.base_model, args.verifier_model)


if __name__ == "__main__":
    main()



