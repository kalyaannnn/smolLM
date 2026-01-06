"""
Compute cost estimation and time projection utilities.

Tracks training progress and estimates:
- Tokens/second throughput
- Tokens/dollar (based on GPU pricing)
- Cost so far and total projected cost
- Time to completion
"""
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


# GPU pricing (USD per hour)
GPU_PRICING = {
    "a100_80gb": 2.50,      # Colab Pro+ / typical cloud pricing
    "a100_40gb": 2.00,
    "h100": 4.00,
    "v100": 1.00,
    "t4": 0.35,
    "a10g": 1.00,
}


@dataclass
class CostTracker:
    """
    Track compute costs and estimate time/money to completion.
    
    Useful for budget planning and demonstrating cost awareness.
    """
    total_tokens: int
    tokens_per_step: int
    gpu_type: str = "a100_80gb"
    hourly_rate: Optional[float] = None  # Override auto pricing
    
    # Internal tracking
    start_time: float = field(default_factory=time.time)
    tokens_processed: int = 0
    steps_completed: int = 0
    
    def __post_init__(self):
        if self.hourly_rate is None:
            self.hourly_rate = GPU_PRICING.get(self.gpu_type, 2.50)
    
    def update(self, tokens: int = None, steps: int = 1):
        """Update progress."""
        if tokens is None:
            tokens = steps * self.tokens_per_step
        self.tokens_processed += tokens
        self.steps_completed += steps
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def elapsed_hours(self) -> float:
        return self.elapsed_seconds / 3600
    
    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds < 1:
            return 0
        return self.tokens_processed / self.elapsed_seconds
    
    @property
    def tokens_per_hour(self) -> float:
        return self.tokens_per_second * 3600
    
    @property
    def tokens_per_dollar(self) -> float:
        if self.hourly_rate <= 0:
            return float('inf')
        return self.tokens_per_hour / self.hourly_rate
    
    @property
    def cost_so_far(self) -> float:
        return self.elapsed_hours * self.hourly_rate
    
    @property
    def progress_fraction(self) -> float:
        if self.total_tokens <= 0:
            return 0
        return min(1.0, self.tokens_processed / self.total_tokens)
    
    @property
    def progress_percent(self) -> float:
        return self.progress_fraction * 100
    
    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_tokens - self.tokens_processed)
    
    @property
    def eta_seconds(self) -> float:
        if self.tokens_per_second <= 0:
            return float('inf')
        return self.remaining_tokens / self.tokens_per_second
    
    @property
    def eta_hours(self) -> float:
        return self.eta_seconds / 3600
    
    @property
    def estimated_total_cost(self) -> float:
        if self.progress_fraction <= 0:
            return 0
        return self.cost_so_far / self.progress_fraction
    
    @property
    def estimated_remaining_cost(self) -> float:
        return max(0, self.estimated_total_cost - self.cost_so_far)
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics as a dictionary."""
        return {
            "tokens_processed": self.tokens_processed,
            "tokens_remaining": self.remaining_tokens,
            "progress_percent": self.progress_percent,
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_dollar": self.tokens_per_dollar,
            "elapsed_hours": self.elapsed_hours,
            "eta_hours": self.eta_hours,
            "cost_so_far_usd": self.cost_so_far,
            "estimated_total_cost_usd": self.estimated_total_cost,
            "estimated_remaining_cost_usd": self.estimated_remaining_cost,
        }
    
    def format_status(self) -> str:
        """Format a human-readable status string."""
        stats = self.get_stats()
        
        lines = [
            f"Progress: {stats['progress_percent']:.1f}%",
            f"Tokens: {stats['tokens_processed']:,} / {self.total_tokens:,}",
            f"Speed: {stats['tokens_per_second']:,.0f} tokens/sec",
            f"Efficiency: {stats['tokens_per_dollar']:,.0f} tokens/$",
            f"Time: {stats['elapsed_hours']:.1f}h elapsed, {stats['eta_hours']:.1f}h remaining",
            f"Cost: ${stats['cost_so_far_usd']:.2f} spent, ${stats['estimated_remaining_cost_usd']:.2f} remaining",
            f"Total estimated: ${stats['estimated_total_cost_usd']:.2f}",
        ]
        
        return "\n".join(lines)
    
    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            "total_tokens": self.total_tokens,
            "tokens_per_step": self.tokens_per_step,
            "tokens_processed": self.tokens_processed,
            "steps_completed": self.steps_completed,
            "start_time": self.start_time,
            "gpu_type": self.gpu_type,
            "hourly_rate": self.hourly_rate,
        }
    
    @classmethod
    def from_state_dict(cls, state: Dict) -> "CostTracker":
        """Restore from checkpoint."""
        tracker = cls(
            total_tokens=state["total_tokens"],
            tokens_per_step=state["tokens_per_step"],
            gpu_type=state.get("gpu_type", "a100_80gb"),
            hourly_rate=state.get("hourly_rate"),
        )
        tracker.tokens_processed = state["tokens_processed"]
        tracker.steps_completed = state["steps_completed"]
        tracker.start_time = state["start_time"]
        return tracker


def estimate_training_cost(
    total_tokens: int,
    tokens_per_second: float,
    gpu_type: str = "a100_80gb",
) -> Dict[str, float]:
    """
    Estimate total training cost before starting.
    
    Args:
        total_tokens: Total tokens to train on
        tokens_per_second: Expected throughput (benchmark first!)
        gpu_type: GPU type for pricing
        
    Returns:
        Dictionary with time and cost estimates
    """
    hourly_rate = GPU_PRICING.get(gpu_type, 2.50)
    
    total_seconds = total_tokens / tokens_per_second
    total_hours = total_seconds / 3600
    total_cost = total_hours * hourly_rate
    
    return {
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "total_hours": total_hours,
        "total_days": total_hours / 24,
        "hourly_rate_usd": hourly_rate,
        "total_cost_usd": total_cost,
        "tokens_per_dollar": total_tokens / total_cost if total_cost > 0 else float('inf'),
    }


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


# Quick test
if __name__ == "__main__":
    print("Testing cost tracker...")
    
    # Create tracker for 10B tokens
    tracker = CostTracker(
        total_tokens=10_000_000_000,
        tokens_per_step=1_572_864,  # 48 * 16 * 2048
        gpu_type="a100_80gb",
    )
    
    # Simulate some progress
    for _ in range(100):
        tracker.update(steps=1)
        time.sleep(0.01)  # Simulate work
    
    print("\nCost tracking status:")
    print(tracker.format_status())
    
    # Test pre-training estimate
    print("\n\nPre-training cost estimate:")
    estimate = estimate_training_cost(
        total_tokens=10_000_000_000,
        tokens_per_second=150_000,  # Typical A100 throughput for 600M model
        gpu_type="a100_80gb",
    )
    for key, value in estimate.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value:,}")
    
    print("\nCost tracker test passed!")



