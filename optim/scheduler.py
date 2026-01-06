"""
WSD (Warmup-Stable-Decay) Learning Rate Scheduler.

A three-phase scheduler commonly used for LLM training:
1. Warmup: Linear increase from 0 to peak LR
2. Stable: Constant at peak LR
3. Decay: Cosine decay to min_lr

This is preferred over pure cosine schedules for longer training runs.
"""
import math
from typing import Optional, List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WSDScheduler(LRScheduler):
    """
    Warmup-Stable-Decay learning rate scheduler.
    
    Schedule:
    - Warmup (steps 0 to warmup_steps): Linear from 0 to base_lr
    - Stable (warmup_steps to total_steps - decay_steps): Constant base_lr
    - Decay (last decay_steps): Cosine decay from base_lr to min_lr
    
    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        decay_ratio: Fraction of training for decay phase (default: 0.1 = last 10%)
        min_lr_ratio: Final LR as fraction of peak (default: 0.1)
        last_epoch: For resuming
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = int(total_steps * decay_ratio)
        self.stable_steps = total_steps - warmup_steps - self.decay_steps
        self.min_lr_ratio = min_lr_ratio
        
        # Validate
        assert warmup_steps >= 0, f"warmup_steps must be >= 0, got {warmup_steps}"
        assert total_steps > warmup_steps, f"total_steps ({total_steps}) must be > warmup_steps ({warmup_steps})"
        assert self.stable_steps >= 0, f"Not enough steps for warmup + decay. stable_steps = {self.stable_steps}"
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase: linear increase
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase: constant LR
            return list(self.base_lrs)
        
        else:
            # Decay phase: cosine decay
            decay_step = step - self.warmup_steps - self.stable_steps
            decay_ratio = decay_step / max(1, self.decay_steps)
            
            # Cosine decay from 1.0 to min_lr_ratio
            cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            lr_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def get_phase(self) -> str:
        """Get current training phase name."""
        step = self.last_epoch
        if step < self.warmup_steps:
            return "warmup"
        elif step < self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"
    
    def get_progress(self) -> dict:
        """Get detailed progress information."""
        step = self.last_epoch
        current_lr = self.get_last_lr()[0] if self.get_last_lr() else self.base_lrs[0]
        
        return {
            "step": step,
            "total_steps": self.total_steps,
            "phase": self.get_phase(),
            "current_lr": current_lr,
            "warmup_progress": min(1.0, step / max(1, self.warmup_steps)),
            "overall_progress": step / self.total_steps,
        }


class WSDSchedulerDual:
    """
    Convenience wrapper for scheduling two optimizers (Muon + AdamW).
    
    Both optimizers follow the same WSD schedule but with different base LRs.
    """
    
    def __init__(
        self,
        muon_optimizer: Optimizer,
        adam_optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        decay_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
    ):
        self.muon_scheduler = WSDScheduler(
            muon_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_ratio=decay_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        self.adam_scheduler = WSDScheduler(
            adam_optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_ratio=decay_ratio,
            min_lr_ratio=min_lr_ratio,
        )
    
    def step(self):
        """Advance both schedulers."""
        self.muon_scheduler.step()
        self.adam_scheduler.step()
    
    def get_last_lr(self) -> dict:
        """Get current LRs for both optimizers."""
        return {
            "muon_lr": self.muon_scheduler.get_last_lr()[0],
            "adam_lr": self.adam_scheduler.get_last_lr()[0],
        }
    
    def get_phase(self) -> str:
        """Get current phase."""
        return self.muon_scheduler.get_phase()
    
    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "muon_scheduler": self.muon_scheduler.state_dict(),
            "adam_scheduler": self.adam_scheduler.state_dict(),
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.muon_scheduler.load_state_dict(state_dict["muon_scheduler"])
        self.adam_scheduler.load_state_dict(state_dict["adam_scheduler"])


def create_wsd_scheduler(
    optimizer: Optimizer,
    total_tokens: int,
    tokens_per_step: int,
    warmup_steps: int = 2000,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
) -> WSDScheduler:
    """
    Create WSD scheduler from token budget.
    
    Args:
        optimizer: Optimizer to schedule
        total_tokens: Total training tokens
        tokens_per_step: Tokens per optimizer step (batch_size * seq_len * grad_accum)
        warmup_steps: Warmup steps
        decay_ratio: Fraction of training for decay
        min_lr_ratio: Final LR ratio
        
    Returns:
        Configured WSD scheduler
    """
    total_steps = total_tokens // tokens_per_step
    
    return WSDScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        decay_ratio=decay_ratio,
        min_lr_ratio=min_lr_ratio,
    )


# Quick test
if __name__ == "__main__":
    import torch
    
    print("Testing WSD Scheduler...")
    
    # Create dummy optimizer
    params = [torch.randn(10, 10, requires_grad=True)]
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    
    # Create scheduler
    scheduler = WSDScheduler(
        optimizer,
        warmup_steps=1000,
        total_steps=10000,
        decay_ratio=0.1,  # Last 10% = 1000 steps for decay
        min_lr_ratio=0.1,
    )
    
    print(f"Total steps: {scheduler.total_steps}")
    print(f"Warmup steps: {scheduler.warmup_steps}")
    print(f"Stable steps: {scheduler.stable_steps}")
    print(f"Decay steps: {scheduler.decay_steps}")
    
    # Test a few steps
    test_steps = [0, 500, 1000, 5000, 9000, 9500, 9999]
    print("\nLR at various steps:")
    for step in test_steps:
        scheduler.last_epoch = step
        lr = scheduler.get_lr()[0]
        phase = scheduler.get_phase()
        print(f"  Step {step:5d}: LR = {lr:.6f}, Phase = {phase}")
    
    print("\nWSD Scheduler test passed!")

