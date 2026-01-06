"""
Training stability monitoring.

Detects issues early to prevent wasted GPU hours:
- Loss spikes
- Gradient explosions/vanishing
- NaN/Inf detection
- Learning rate anomalies
"""
import math
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

import torch


@dataclass
class StabilityAlert:
    """A detected training stability issue."""
    step: int
    severity: str  # "warning", "error", "critical"
    category: str  # "loss_spike", "gradient", "nan", "lr"
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


class StabilityMonitor:
    """
    Monitor training stability and detect issues early.
    
    Tracks:
    - Loss history and spike detection
    - Gradient norm history and explosion detection
    - NaN/Inf in loss or gradients
    - Learning rate anomalies
    """
    
    def __init__(
        self,
        loss_spike_threshold: float = 2.0,    # Alert if loss > 2x rolling average
        grad_norm_threshold: float = 100.0,    # Alert if grad norm > threshold
        grad_norm_spike_threshold: float = 5.0,  # Alert if grad norm > 5x rolling avg
        window_size: int = 100,                # Window for rolling statistics
        loss_increase_patience: int = 50,      # Alert after N steps of increasing loss
    ):
        self.loss_spike_threshold = loss_spike_threshold
        self.grad_norm_threshold = grad_norm_threshold
        self.grad_norm_spike_threshold = grad_norm_spike_threshold
        self.window_size = window_size
        self.loss_increase_patience = loss_increase_patience
        
        # History tracking
        self.loss_history: deque = deque(maxlen=window_size)
        self.grad_norm_history: deque = deque(maxlen=window_size)
        self.lr_history: deque = deque(maxlen=window_size)
        
        # Trend tracking
        self._loss_increase_count = 0
        self._min_loss_seen = float('inf')
        
        # Alerts
        self.alerts: List[StabilityAlert] = []
    
    def check(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: Optional[float] = None,
    ) -> List[StabilityAlert]:
        """
        Check for stability issues.
        
        Args:
            step: Current training step
            loss: Current loss value
            grad_norm: Current gradient norm
            lr: Current learning rate (optional)
            
        Returns:
            List of any detected alerts
        """
        alerts = []
        
        # Check for NaN/Inf
        if math.isnan(loss) or math.isinf(loss):
            alerts.append(StabilityAlert(
                step=step,
                severity="critical",
                category="nan",
                message=f"Loss is {loss}! Training is diverging.",
                value=loss,
            ))
        
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            alerts.append(StabilityAlert(
                step=step,
                severity="critical",
                category="nan",
                message=f"Gradient norm is {grad_norm}! Training is diverging.",
                value=grad_norm,
            ))
        
        # Only continue checks if we have valid values
        if math.isnan(loss) or math.isinf(loss):
            return alerts
        
        # Check for loss spike
        if len(self.loss_history) >= 10:
            rolling_avg = sum(self.loss_history) / len(self.loss_history)
            if loss > rolling_avg * self.loss_spike_threshold:
                alerts.append(StabilityAlert(
                    step=step,
                    severity="warning",
                    category="loss_spike",
                    message=f"Loss spike detected: {loss:.4f} vs rolling avg {rolling_avg:.4f}",
                    value=loss,
                    threshold=rolling_avg * self.loss_spike_threshold,
                ))
        
        # Check for gradient explosion
        if grad_norm > self.grad_norm_threshold:
            alerts.append(StabilityAlert(
                step=step,
                severity="warning",
                category="gradient",
                message=f"High gradient norm: {grad_norm:.2f} > {self.grad_norm_threshold}",
                value=grad_norm,
                threshold=self.grad_norm_threshold,
            ))
        
        # Check for gradient spike
        if len(self.grad_norm_history) >= 10:
            grad_avg = sum(self.grad_norm_history) / len(self.grad_norm_history)
            if grad_norm > grad_avg * self.grad_norm_spike_threshold:
                alerts.append(StabilityAlert(
                    step=step,
                    severity="warning",
                    category="gradient",
                    message=f"Gradient spike: {grad_norm:.2f} vs rolling avg {grad_avg:.2f}",
                    value=grad_norm,
                    threshold=grad_avg * self.grad_norm_spike_threshold,
                ))
        
        # Check for sustained loss increase
        if loss < self._min_loss_seen:
            self._min_loss_seen = loss
            self._loss_increase_count = 0
        else:
            self._loss_increase_count += 1
        
        if self._loss_increase_count >= self.loss_increase_patience:
            alerts.append(StabilityAlert(
                step=step,
                severity="warning",
                category="loss_spike",
                message=f"Loss not improving for {self._loss_increase_count} steps. Min seen: {self._min_loss_seen:.4f}, current: {loss:.4f}",
                value=loss,
            ))
            self._loss_increase_count = 0  # Reset after alert
        
        # Update history
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        if lr is not None:
            self.lr_history.append(lr)
        
        # Store and return alerts
        self.alerts.extend(alerts)
        return alerts
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        stats = {}
        
        if self.loss_history:
            stats["loss_rolling_avg"] = sum(self.loss_history) / len(self.loss_history)
            stats["loss_rolling_min"] = min(self.loss_history)
            stats["loss_rolling_max"] = max(self.loss_history)
        
        if self.grad_norm_history:
            stats["grad_norm_rolling_avg"] = sum(self.grad_norm_history) / len(self.grad_norm_history)
            stats["grad_norm_rolling_max"] = max(self.grad_norm_history)
        
        stats["min_loss_seen"] = self._min_loss_seen
        stats["steps_since_improvement"] = self._loss_increase_count
        stats["total_alerts"] = len(self.alerts)
        
        return stats
    
    def reset(self):
        """Reset tracking state."""
        self.loss_history.clear()
        self.grad_norm_history.clear()
        self.lr_history.clear()
        self._loss_increase_count = 0
        self._min_loss_seen = float('inf')
        self.alerts.clear()


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return math.sqrt(total_norm)


def clip_gradient_norm(
    model: torch.nn.Module,
    max_norm: float,
) -> float:
    """Clip gradients and return the original norm."""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    )
    return total_norm.item()


def check_for_nan_params(model: torch.nn.Module) -> List[str]:
    """Check for NaN values in model parameters."""
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    return nan_params


def check_for_nan_grads(model: torch.nn.Module) -> List[str]:
    """Check for NaN values in gradients."""
    nan_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_grads.append(name)
    return nan_grads


# Quick test
if __name__ == "__main__":
    print("Testing stability monitor...")
    
    monitor = StabilityMonitor(
        loss_spike_threshold=2.0,
        grad_norm_threshold=100.0,
        window_size=20,
    )
    
    # Simulate normal training
    print("Simulating normal training...")
    for step in range(50):
        loss = 3.0 - step * 0.01 + (0.1 if step % 10 == 0 else 0)
        grad_norm = 5.0 + step * 0.05
        alerts = monitor.check(step, loss, grad_norm)
        if alerts:
            for alert in alerts:
                print(f"  Step {step}: [{alert.severity}] {alert.message}")
    
    # Simulate loss spike
    print("\nSimulating loss spike...")
    alerts = monitor.check(51, 10.0, 5.0)  # Big loss spike
    for alert in alerts:
        print(f"  [{alert.severity}] {alert.message}")
    
    # Simulate gradient explosion
    print("\nSimulating gradient explosion...")
    alerts = monitor.check(52, 2.5, 500.0)  # High grad norm
    for alert in alerts:
        print(f"  [{alert.severity}] {alert.message}")
    
    # Simulate NaN
    print("\nSimulating NaN loss...")
    alerts = monitor.check(53, float('nan'), 5.0)
    for alert in alerts:
        print(f"  [{alert.severity}] {alert.message}")
    
    # Print stats
    print("\nStats:", monitor.get_stats())
    print(f"Total alerts: {len(monitor.alerts)}")
    
    print("\nStability monitor test passed!")

