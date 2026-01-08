#!/usr/bin/env python3
"""
SmolLM Pretraining Script

Pretrains a ~600M parameter language model with:
- Muon + AdamW optimizer split
- WSD learning rate schedule
- Streaming data with domain mixing
- Sequence packing with document masks
- Automatic checkpointing to Google Drive
- W&B logging with JSONL fallback

Usage:
    # Start fresh
    python train_pretrain.py --config configs/pretrain.yaml
    
    # Resume from latest checkpoint
    python train_pretrain.py --config configs/pretrain.yaml --resume
    
    # Resume from specific checkpoint
    python train_pretrain.py --config configs/pretrain.yaml --resume-from path/to/checkpoint.pt
"""
import os
import sys
import math
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.config import SmolLMConfig
from models.transformer import SmolLM
from optim.muon import Muon
from optim.param_groups import get_param_groups
from optim.scheduler import WSDScheduler
from data.tokenizer import load_tokenizer
from data.streaming import MixedDomainDataset, DomainConfig, DEFAULT_DOMAIN_MIX
from data.packing import PackingCollator, PackedBatch
from utils.logging import Logger, MetricTracker, format_metrics
from utils.checkpoint import CheckpointManager
from utils.cost_tracker import CostTracker
from train.stability import StabilityMonitor, clip_gradient_norm
from eval.validation import run_validation


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        # Model
        "model": {
            "d_model": 1536,
            "n_layers": 24,
            "n_heads": 12,
            "n_kv_heads": 4,
            "ffn_dim": 6144,
            "vocab_size": 32000,
            "max_seq_len": 2048,
            "rope_theta": 10000.0,
            "nope_layer_interval": 4,
            "dropout": 0.0,
            "use_flash_attn": True,
        },
        
        # Tokenizer
        "tokenizer": {
            "name": "meta-llama/Llama-2-7b-hf",
        },
        
        # Training
        "training": {
            "total_tokens": 10_000_000_000,  # 10B tokens
            "micro_batch_size": 48,
            "seq_len": 2048,
            "gradient_accumulation": 16,
            "precision": "bf16",
        },
        
        # Optimizer
        "optimizer": {
            "muon_lr": 0.02,
            "adam_lr": 3e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "muon_momentum": 0.95,
            "adam_betas": [0.9, 0.95],
        },
        
        # Scheduler
        "scheduler": {
            "warmup_steps": 2000,
            "decay_ratio": 0.1,
            "min_lr_ratio": 0.1,
        },
        
        # Logging
        "logging": {
            "log_every_steps": 10,
            "eval_every_steps": 100,
            "project": "smol-lm",
            "use_wandb": True,
        },
        
        # Checkpointing
        "checkpoint": {
            "save_every_steps": 500,
            "save_every_hours": 2.0,
            "checkpoint_dir": "/content/drive/MyDrive/smol-lm-checkpoints",
            "keep_last_n": 3,
        },
    }
    
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
        # Merge user config over defaults
        for section, values in user_config.items():
            if section in default_config and isinstance(values, dict):
                default_config[section].update(values)
            else:
                default_config[section] = values
    
    return default_config


def setup_model(config: Dict[str, Any], device: str = "cuda") -> SmolLM:
    """Initialize model with proper configuration."""
    model_config = SmolLMConfig(**config["model"])
    model = SmolLM(model_config)
    model = model.to(device)
    
    # Set precision
    if config["training"]["precision"] == "bf16":
        model = model.bfloat16()
    elif config["training"]["precision"] == "fp16":
        model = model.half()
    
    return model


def setup_optimizers(
    model: SmolLM,
    config: Dict[str, Any],
) -> tuple:
    """Set up Muon + AdamW optimizers with proper parameter grouping."""
    opt_config = config["optimizer"]
    
    # Get parameter groups
    muon_groups, adam_groups = get_param_groups(
        model,
        muon_lr=opt_config["muon_lr"],
        adam_lr=opt_config["adam_lr"],
        weight_decay=opt_config["weight_decay"],
        verbose=True,
    )
    
    # Create optimizers
    muon_optimizer = Muon(
        muon_groups,
        lr=opt_config["muon_lr"],
        momentum=opt_config["muon_momentum"],
    )
    
    adam_optimizer = torch.optim.AdamW(
        adam_groups,
        lr=opt_config["adam_lr"],
        betas=tuple(opt_config["adam_betas"]),
        eps=1e-8,
    )
    
    return muon_optimizer, adam_optimizer


def setup_scheduler(
    muon_optimizer: Muon,
    adam_optimizer: torch.optim.AdamW,
    config: Dict[str, Any],
) -> tuple:
    """Set up WSD schedulers for both optimizers."""
    train_config = config["training"]
    sched_config = config["scheduler"]
    
    # Calculate total steps
    tokens_per_step = (
        train_config["micro_batch_size"] *
        train_config["seq_len"] *
        train_config["gradient_accumulation"]
    )
    total_steps = train_config["total_tokens"] // tokens_per_step
    
    muon_scheduler = WSDScheduler(
        muon_optimizer,
        warmup_steps=sched_config["warmup_steps"],
        total_steps=total_steps,
        decay_ratio=sched_config["decay_ratio"],
        min_lr_ratio=sched_config["min_lr_ratio"],
    )
    
    adam_scheduler = WSDScheduler(
        adam_optimizer,
        warmup_steps=sched_config["warmup_steps"],
        total_steps=total_steps,
        decay_ratio=sched_config["decay_ratio"],
        min_lr_ratio=sched_config["min_lr_ratio"],
    )
    
    return muon_scheduler, adam_scheduler, total_steps


def create_dataloader(
    tokenizer,
    config: Dict[str, Any],
):
    """Create streaming dataloader with packing."""
    train_config = config["training"]
    data_config = config.get("data", {})

    # Build domain configs from YAML (fallback to defaults if missing)
    domain_configs = {}
    if data_config:
        for name, cfg in data_config.items():
            sources = [(s["name"], s["weight"]) for s in cfg.get("sources", [])]
            default_domain = DEFAULT_DOMAIN_MIX.get(name)
            text_field = cfg.get("text_field") or (default_domain.text_field if default_domain else "text")
            domain_configs[name] = DomainConfig(
                name=name,
                weight=cfg.get("weight", 0),
                sources=sources,
                max_tokens=cfg.get("max_tokens"),
                text_field=text_field,
            )
    else:
        domain_configs = DEFAULT_DOMAIN_MIX
    
    # Create dataset
    dataset = MixedDomainDataset(
        tokenizer=tokenizer,
        domains=domain_configs,
        seed=42,
        total_tokens=train_config["total_tokens"],
        max_length=train_config["seq_len"],
    )
    
    # Create collator for packing
    collator = PackingCollator(
        seq_len=train_config["seq_len"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config["micro_batch_size"],
        num_workers=0,  # Streaming doesn't work well with workers
        collate_fn=collator,
    )
    
    return dataloader, dataset


def train_step(
    model: SmolLM,
    batch: PackedBatch,
    muon_optimizer: Muon,
    adam_optimizer: torch.optim.AdamW,
    scaler: Optional[GradScaler],
    grad_clip: float,
    accumulation_steps: int,
    current_accumulation: int,
    device: str = "cuda",
    precision: str = "bf16",
) -> Dict[str, float]:
    """Execute a single training step."""
    # Move batch to device
    input_ids = batch.input_ids.to(device)
    labels = batch.labels.to(device)
    attention_mask = batch.attention_mask.to(device)
    position_ids = batch.position_ids.to(device)
    
    # Determine dtype
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    
    # Forward pass with autocast
    with autocast(dtype=dtype):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        loss = outputs["loss"] / accumulation_steps
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    
    # Only step optimizers after accumulation
    metrics = {"loss": loss.item() * accumulation_steps}
    
    if (current_accumulation + 1) % accumulation_steps == 0:
        # Unscale for gradient clipping
        if scaler is not None:
            scaler.unscale_(muon_optimizer)
            scaler.unscale_(adam_optimizer)
        
        # Clip gradients
        grad_norm = clip_gradient_norm(model, grad_clip)
        metrics["grad_norm"] = grad_norm
        
        # Optimizer step
        if scaler is not None:
            scaler.step(muon_optimizer)
            scaler.step(adam_optimizer)
            scaler.update()
        else:
            muon_optimizer.step()
            adam_optimizer.step()
        
        # Zero gradients
        muon_optimizer.zero_grad()
        adam_optimizer.zero_grad()
    
    return metrics


def train(
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
):
    """Main training loop."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_config = config["training"]
    opt_config = config["optimizer"]
    
    # Calculate tokens per step
    tokens_per_step = (
        train_config["micro_batch_size"] *
        train_config["seq_len"] *
        train_config["gradient_accumulation"]
    )
    
    # Initialize run name
    run_name = f"smol_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
        save_every_hours=config["checkpoint"]["save_every_hours"],
        keep_last_n=config["checkpoint"]["keep_last_n"],
        run_name=run_name,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config["tokenizer"]["name"])
    config["model"]["vocab_size"] = len(tokenizer)
    
    # Setup model
    print("Initializing model...")
    model = setup_model(config, device)
    
    # Setup optimizers
    print("Setting up optimizers...")
    muon_optimizer, adam_optimizer = setup_optimizers(model, config)
    
    # Setup schedulers
    muon_scheduler, adam_scheduler, total_steps = setup_scheduler(
        muon_optimizer, adam_optimizer, config
    )
    
    # Setup gradient scaler (only for FP16)
    scaler = GradScaler() if train_config["precision"] == "fp16" else None
    
    # Setup cost tracker
    cost_tracker = CostTracker(
        total_tokens=train_config["total_tokens"],
        tokens_per_step=tokens_per_step,
    )
    
    # Setup stability monitor
    stability_monitor = StabilityMonitor()
    
    # Setup metric tracker
    metric_tracker = MetricTracker(window_size=100)
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        checkpoint = checkpoint_manager.load(
            resume_from,
            model,
            {"muon": muon_optimizer, "adam": adam_optimizer},
            device=device,
        )
        start_step = checkpoint["step"]
        if checkpoint.get("scheduler_state"):
            # Manually set scheduler step
            muon_scheduler.last_epoch = start_step
            adam_scheduler.last_epoch = start_step
        cost_tracker.tokens_processed = start_step * tokens_per_step
        cost_tracker.steps_completed = start_step
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader, dataset = create_dataloader(tokenizer, config)
    
    # Training loop
    print(f"\nStarting training from step {start_step}")
    print(f"Total steps: {total_steps}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total tokens: {train_config['total_tokens']:,}")
    
    model.train()
    step = start_step
    accumulation_count = 0
    
    start_time = time.time()
    step_start_time = start_time
    
    try:
        for batch in dataloader:
            # Check if we've completed training
            if step >= total_steps:
                break
            
            # Training step
            metrics = train_step(
                model=model,
                batch=batch,
                muon_optimizer=muon_optimizer,
                adam_optimizer=adam_optimizer,
                scaler=scaler,
                grad_clip=opt_config["grad_clip"],
                accumulation_steps=train_config["gradient_accumulation"],
                current_accumulation=accumulation_count,
                device=device,
                precision=train_config["precision"],
            )
            
            accumulation_count += 1
            
            # Only log and step after full accumulation
            if accumulation_count % train_config["gradient_accumulation"] != 0:
                continue
            
            # Update schedulers
            muon_scheduler.step()
            adam_scheduler.step()
            
            # Update cost tracker
            cost_tracker.update(steps=1)
            
            # Track metrics
            metric_tracker.update("loss", metrics["loss"])
            if "grad_norm" in metrics:
                metric_tracker.update("grad_norm", metrics["grad_norm"])
            
            # Check stability
            alerts = stability_monitor.check(
                step=step,
                loss=metrics["loss"],
                grad_norm=metrics.get("grad_norm", 0),
                lr=muon_scheduler.get_last_lr()[0],
            )
            for alert in alerts:
                logger.alert(alert.category, alert.message, alert.severity.upper())
            
            # Logging
            if step % config["logging"]["log_every_steps"] == 0:
                step_time = time.time() - step_start_time
                tokens_per_sec = tokens_per_step / max(step_time, 0.001)
                
                log_metrics = {
                    "train/loss": metrics["loss"],
                    "train/grad_norm": metrics.get("grad_norm", 0),
                    "train/muon_lr": muon_scheduler.get_last_lr()[0],
                    "train/adam_lr": adam_scheduler.get_last_lr()[0],
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_processed": cost_tracker.tokens_processed,
                    "train/progress_percent": cost_tracker.progress_percent,
                    "cost/usd_spent": cost_tracker.cost_so_far,
                    "cost/tokens_per_dollar": cost_tracker.tokens_per_dollar,
                    "cost/eta_hours": cost_tracker.eta_hours,
                }
                
                # Log GPU memory
                if torch.cuda.is_available():
                    log_metrics["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                    log_metrics["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                
                logger.log(log_metrics, step=step)
                
                # Print progress
                print(f"Step {step}/{total_steps} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"LR: {muon_scheduler.get_last_lr()[0]:.6f} | "
                      f"Tokens/s: {tokens_per_sec:,.0f} | "
                      f"Progress: {cost_tracker.progress_percent:.1f}%")
                
                step_start_time = time.time()
            
            # Validation (periodic)
            if step > 0 and step % config["logging"]["eval_every_steps"] == 0:
                print(f"\nRunning validation at step {step}...")
                try:
                    val_metrics = run_validation(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        device=device,
                        max_samples_per_domain=100,  # Limit for speed
                    )
                    
                    # Log validation metrics
                    logger.log(val_metrics, step=step)
                    
                    # Print validation results
                    print(f"Validation - Loss: {val_metrics.get('val_loss', 0):.4f}, "
                          f"PPL: {val_metrics.get('val_ppl', 0):.2f}")
                    if "web_val_ppl" in val_metrics:
                        print(f"  Web PPL: {val_metrics['web_val_ppl']:.2f}, "
                              f"Code PPL: {val_metrics.get('code_val_ppl', 0):.2f}, "
                              f"Math PPL: {val_metrics.get('math_val_ppl', 0):.2f}")
                except Exception as e:
                    print(f"Validation failed: {e}")
            
            # Checkpointing
            if checkpoint_manager.should_save(step):
                checkpoint_manager.save(
                    step=step,
                    model=model,
                    optimizers={"muon": muon_optimizer, "adam": adam_optimizer},
                    scheduler={"muon": muon_scheduler.state_dict(), "adam": adam_scheduler.state_dict()},
                    config=config,
                    metrics=metric_tracker.get_all_means(),
                    dataloader_state=dataset.get_state(),
                )
            
            step += 1
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        # Save emergency checkpoint
        checkpoint_manager.save(
            step=step,
            model=model,
            optimizers={"muon": muon_optimizer, "adam": adam_optimizer},
            scheduler={"muon": muon_scheduler.state_dict(), "adam": adam_scheduler.state_dict()},
            config=config,
            metrics=metric_tracker.get_all_means(),
        )
        raise
    
    finally:
        # Save final checkpoint
        checkpoint_manager.save_final(model, config, stage="pretrain")
        
        # Log final stats
        total_time = time.time() - start_time
        print(f"\nTraining completed!")
        print(f"Total time: {total_time / 3600:.2f} hours")
        print(f"Total tokens: {cost_tracker.tokens_processed:,}")
        final_loss = metric_tracker.get_last("loss")
        final_loss_str = f"{final_loss:.4f}" if final_loss is not None else "n/a"
        print(f"Final loss: {final_loss_str}")
        print(cost_tracker.format_status())
        
        logger.finish()


def main():
    parser = argparse.ArgumentParser(description="SmolLM Pretraining")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Handle resume
    resume_from = args.resume_from
    if args.resume and not resume_from:
        # Find latest checkpoint
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=config["checkpoint"]["checkpoint_dir"]
        )
        latest = checkpoint_manager.get_latest_checkpoint()
        if latest:
            resume_from = str(latest)
            print(f"Resuming from: {resume_from}")
    
    # Start training
    train(config, resume_from=resume_from)


if __name__ == "__main__":
    main()
