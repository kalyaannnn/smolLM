"""
Logging utilities with W&B integration and JSONL fallback.

Dual logging ensures metrics are never lost:
- Primary: Weights & Biases for rich visualization
- Fallback: Local JSONL file for reliability
"""
import json
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class Logger:
    """
    Dual logger with W&B + JSONL fallback.
    
    Always logs to local JSONL for reliability.
    Optionally logs to W&B for visualization.
    """
    
    def __init__(
        self,
        project: str = "smol-lm",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_dir: str = "./logs",
        use_wandb: bool = True,
        wandb_entity: Optional[str] = None,
    ):
        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        self.log_dir = Path(log_dir)
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize JSONL file
        self.jsonl_path = self.log_dir / f"{self.run_name}.jsonl"
        
        # Initialize W&B if available
        self.wandb_run = None
        if self.use_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=project,
                    name=self.run_name,
                    config=config,
                    entity=wandb_entity,
                    resume="allow",
                )
                print(f"W&B initialized: {wandb.run.url}")
            except Exception as e:
                print(f"W&B init failed, using JSONL only: {e}")
                self.use_wandb = False
        
        # Track step
        self._step = 0
        self._start_time = time.time()
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B and JSONL.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Training step (auto-incremented if None)
            commit: Whether to commit to W&B immediately
        """
        if step is None:
            step = self._step
            self._step += 1
        
        # Add timestamp and step
        log_entry = {
            "step": step,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self._start_time,
            **metrics,
        }
        
        # Log to JSONL (always)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            try:
                wandb.log(metrics, step=step, commit=commit)
            except Exception as e:
                print(f"W&B log failed: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.config.update(config)
        
        # Save to JSONL
        config_path = self.log_dir / f"{self.run_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Update W&B config
        if self.use_wandb and self.wandb_run:
            wandb.config.update(config, allow_val_change=True)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log summary metrics (shown in W&B overview)."""
        if self.use_wandb and self.wandb_run:
            for key, value in summary.items():
                wandb.run.summary[key] = value
    
    def log_artifact(
        self,
        path: str,
        name: str,
        artifact_type: str = "model",
    ):
        """Log an artifact (model checkpoint, etc) to W&B."""
        if self.use_wandb and self.wandb_run:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            wandb.log_artifact(artifact)
    
    def alert(self, title: str, text: str, level: str = "WARN"):
        """Send an alert (useful for training issues)."""
        print(f"[{level}] {title}: {text}")
        
        if self.use_wandb and self.wandb_run:
            try:
                wandb.alert(
                    title=title,
                    text=text,
                    level=getattr(wandb.AlertLevel, level, wandb.AlertLevel.WARN),
                )
            except Exception:
                pass
    
    def finish(self):
        """Finish logging (call at end of training)."""
        if self.use_wandb and self.wandb_run:
            wandb.finish()
        
        print(f"Logs saved to: {self.jsonl_path}")
    
    @property
    def step(self) -> int:
        return self._step
    
    @step.setter
    def step(self, value: int):
        self._step = value


class MetricTracker:
    """
    Track and compute running statistics for metrics.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: Dict[str, list] = {}
    
    def update(self, name: str, value: float):
        """Add a value to tracking."""
        if name not in self._values:
            self._values[name] = []
        
        self._values[name].append(value)
        
        # Keep only last window_size values
        if len(self._values[name]) > self.window_size:
            self._values[name] = self._values[name][-self.window_size:]
    
    def get_mean(self, name: str) -> Optional[float]:
        """Get mean of tracked values."""
        if name not in self._values or not self._values[name]:
            return None
        return sum(self._values[name]) / len(self._values[name])
    
    def get_last(self, name: str) -> Optional[float]:
        """Get most recent value."""
        if name not in self._values or not self._values[name]:
            return None
        return self._values[name][-1]
    
    def get_all_means(self) -> Dict[str, float]:
        """Get means for all tracked metrics."""
        return {
            name: self.get_mean(name)
            for name in self._values
            if self.get_mean(name) is not None
        }
    
    def reset(self, name: Optional[str] = None):
        """Reset tracking for a metric or all metrics."""
        if name:
            self._values.pop(name, None)
        else:
            self._values.clear()


def format_metrics(metrics: Dict[str, Any], precision: int = 4) -> str:
    """Format metrics for printing."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.{precision}f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


# Quick test
if __name__ == "__main__":
    print("Testing logging utilities...")
    
    # Create logger (W&B disabled for test)
    logger = Logger(
        project="test",
        run_name="test_run",
        use_wandb=False,
        log_dir="./test_logs",
    )
    
    # Log some metrics
    for step in range(10):
        logger.log({
            "loss": 1.0 / (step + 1),
            "lr": 0.001 * (1 - step / 10),
        }, step=step)
    
    # Log config
    logger.log_config({"model_size": "600M", "batch_size": 32})
    
    logger.finish()
    
    # Test metric tracker
    tracker = MetricTracker(window_size=5)
    for i in range(10):
        tracker.update("loss", 1.0 / (i + 1))
    
    print(f"Mean loss (last 5): {tracker.get_mean('loss'):.4f}")
    print(f"Last loss: {tracker.get_last('loss'):.4f}")
    
    # Cleanup
    import shutil
    shutil.rmtree("./test_logs", ignore_errors=True)
    
    print("\nLogging test passed!")



