"""
Checkpoint manager with Google Drive support for Colab reliability.

Features:
- Automatic checkpointing (by step count and time)
- Exact resume with optimizer/scheduler/RNG states
- Atomic writes to prevent corruption
- Google Drive integration for Colab
- Checkpoint rotation (keep last N)
"""
import os
import time
import random
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch
import numpy as np


class CheckpointManager:
    """
    Manages training checkpoints with Colab reliability in mind.
    
    Saves to Google Drive (or local path) with atomic writes.
    Supports exact training resume including RNG states.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "/content/drive/MyDrive/smol-lm-checkpoints",
        save_every_steps: int = 500,
        save_every_hours: float = 2.0,
        keep_last_n: int = 3,
        run_name: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_steps = save_every_steps
        self.save_every_seconds = save_every_hours * 3600
        self.keep_last_n = keep_last_n
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self._last_save_time = time.time()
        self._last_save_step = 0
    
    def should_save(self, step: int) -> bool:
        """Check if we should save a checkpoint."""
        # Check step interval
        if step - self._last_save_step >= self.save_every_steps:
            return True
        
        # Check time interval
        if time.time() - self._last_save_time >= self.save_every_seconds:
            return True
        
        return False
    
    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizers: Dict[str, torch.optim.Optimizer],
        scheduler: Any,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        dataloader_state: Optional[Dict] = None,
        extra_state: Optional[Dict] = None,
    ) -> str:
        """
        Save a complete checkpoint for exact resume.
        
        Args:
            step: Current training step
            model: The model
            optimizers: Dict of optimizer name -> optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            metrics: Current metrics
            dataloader_state: Dataloader state for deterministic resume
            extra_state: Any additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            # Training state
            "step": step,
            "config": config,
            "metrics": metrics or {},
            
            # Model state
            "model_state": model.state_dict(),
            
            # Optimizer states
            "optimizer_states": {
                name: opt.state_dict() for name, opt in optimizers.items()
            },
            
            # Scheduler state
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            
            # Dataloader state for deterministic resume
            "dataloader_state": dataloader_state,
            
            # RNG states for exact reproducibility
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            
            # Extra state
            "extra_state": extra_state or {},
            
            # Metadata
            "timestamp": datetime.now().isoformat(),
            "run_name": self.run_name,
        }
        
        # Generate filename
        filename = f"{self.run_name}_step_{step:08d}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Atomic write (save to temp, then rename)
        temp_path = filepath.with_suffix(".tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(filepath)
        
        # Update tracking
        self._last_save_time = time.time()
        self._last_save_step = step
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
        
        print(f"Checkpoint saved: {filepath}")
        return str(filepath)
    
    def _rotate_checkpoints(self):
        """Keep only the last N checkpoints."""
        if self.keep_last_n <= 0:
            return
        
        # Find all checkpoints for this run
        pattern = f"{self.run_name}_step_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        # Delete old ones
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            old.unlink()
            print(f"Deleted old checkpoint: {old}")
    
    def load_latest(
        self,
        model: torch.nn.Module,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load weights into
            optimizers: Optimizers to restore state
            scheduler: Scheduler to restore state
            device: Device to load to
            
        Returns:
            Checkpoint dict if found, None otherwise
        """
        checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoint found, starting fresh")
            return None
        
        return self.load(checkpoint_path, model, optimizers, scheduler, device)
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        # Check for run-specific checkpoints first
        pattern = f"{self.run_name}_step_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        if checkpoints:
            return checkpoints[-1]
        
        # Fall back to any checkpoint
        all_checkpoints = sorted(self.checkpoint_dir.glob("*_step_*.pt"))
        if all_checkpoints:
            return all_checkpoints[-1]
        
        return None
    
    def load(
        self,
        path: str,
        model: torch.nn.Module,
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """
        Load a specific checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Model to load weights into
            optimizers: Optimizers to restore
            scheduler: Scheduler to restore
            device: Device to load to
            
        Returns:
            Checkpoint dict with restored state
        """
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        
        # Load model
        model.load_state_dict(checkpoint["model_state"])
        
        # Load optimizers
        if optimizers and "optimizer_states" in checkpoint:
            for name, opt in optimizers.items():
                if name in checkpoint["optimizer_states"]:
                    opt.load_state_dict(checkpoint["optimizer_states"][name])
        
        # Load scheduler
        if scheduler and checkpoint.get("scheduler_state"):
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        # Restore RNG states
        if "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        
        # Update tracking
        self._last_save_step = checkpoint["step"]
        self._last_save_time = time.time()
        
        print(f"Resumed from step {checkpoint['step']}")
        return checkpoint
    
    def save_final(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        stage: str = "pretrain",
    ) -> str:
        """
        Save final checkpoint for a training stage.
        
        This creates a named checkpoint (e.g., pretrain_final.pt)
        that can be used as input for the next stage.
        """
        checkpoint = {
            "config": config,
            "model_state": model.state_dict(),
            "metrics": metrics or {},
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
        }
        
        filename = f"{stage}_final.pt"
        filepath = self.checkpoint_dir / filename
        
        torch.save(checkpoint, filepath)
        print(f"Final {stage} checkpoint saved: {filepath}")
        
        return str(filepath)


def save_checkpoint_simple(
    path: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    step: Optional[int] = None,
):
    """Simple checkpoint save (model + config only)."""
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "step": step,
    }, path)


def load_checkpoint_simple(
    path: str,
    model: torch.nn.Module,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Simple checkpoint load."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


# Quick test
if __name__ == "__main__":
    print("Testing checkpoint manager...")
    
    # Create test model and optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir="./test_checkpoints",
        save_every_steps=10,
        keep_last_n=2,
        run_name="test",
    )
    
    # Save checkpoint
    path = manager.save(
        step=100,
        model=model,
        optimizers={"main": optimizer},
        scheduler=None,
        config={"test": True},
        metrics={"loss": 0.5},
    )
    
    # Load checkpoint
    new_model = torch.nn.Linear(10, 10)
    new_optimizer = torch.optim.Adam(new_model.parameters())
    
    checkpoint = manager.load(
        path,
        new_model,
        {"main": new_optimizer},
        device="cpu",
    )
    
    print(f"Loaded step: {checkpoint['step']}")
    print(f"Loaded metrics: {checkpoint['metrics']}")
    
    # Cleanup
    shutil.rmtree("./test_checkpoints", ignore_errors=True)
    
    print("\nCheckpoint test passed!")



