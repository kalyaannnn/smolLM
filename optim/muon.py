"""
Muon Optimizer - Momentum Orthogonalized Update.

Muon applies Newton-Schulz orthogonalization to momentum updates for
2D weight matrices, which can converge faster than Adam for these params.

Ported from modded-nanogpt with modifications for production use.
Reference: https://github.com/KellerJordan/modded-nanogpt
"""
import math
from typing import List, Optional, Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute G @ (G.T @ G)^{-1/2}.
    
    This orthogonalizes the columns of G, which helps optimization
    by removing correlations in gradient directions.
    
    Args:
        G: Input matrix [fan_out, fan_in]
        steps: Number of Newton-Schulz iterations (5 is usually enough)
        
    Returns:
        Orthogonalized matrix
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Optimal coefficients for 5 iterations
    
    X = G.bfloat16() if G.dtype == torch.float32 else G
    
    # Normalize for numerical stability
    X = X / (X.norm() + 1e-7)
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon optimizer for 2D weight matrices.
    
    Uses Newton-Schulz orthogonalization on momentum updates, which
    can lead to faster convergence on weight matrices compared to Adam.
    
    Args:
        params: Parameters to optimize (should be 2D weight matrices)
        lr: Learning rate (default: 0.02)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Newton-Schulz iterations (default: 5)
        
    Note:
        - Only use for 2D weight matrices (attention Q/K/V/O, MLP weights)
        - For biases, norms, and embeddings, use AdamW instead
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Only apply to 2D tensors
                if grad.ndim != 2:
                    # Fallback to SGD with momentum for non-2D params
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                    
                    p.add_(grad, alpha=-lr)
                    continue
                
                # Get state
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                # Update momentum buffer
                buf.mul_(momentum).add_(grad)
                
                # Apply Nesterov momentum
                if nesterov:
                    update = grad.add(buf, alpha=momentum)
                else:
                    update = buf.clone()
                
                # Apply Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                
                # Scale by sqrt of fan_out (larger matrices need smaller updates)
                scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                
                # Update parameters
                p.add_(update, alpha=-lr * scale)
        
        return loss


class MuonWithBackup(Optimizer):
    """
    Muon optimizer with automatic fallback to SGD+momentum for non-2D params.
    
    This is a convenience wrapper that handles mixed parameter types.
    For production, prefer separate Muon + AdamW optimizers with explicit
    parameter grouping (see param_groups.py).
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        backup_lr: float = 3e-4,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            backup_lr=backup_lr,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            backup_lr = group['backup_lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['is_2d'] = grad.ndim == 2
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                if state['is_2d']:
                    # Muon update for 2D
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf.clone()
                    
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                    p.add_(update, alpha=-lr * scale)
                else:
                    # SGD+momentum for non-2D
                    if nesterov:
                        update = grad.add(buf, alpha=momentum)
                    else:
                        update = buf
                    p.add_(update, alpha=-backup_lr)
        
        return loss


# Quick test
if __name__ == "__main__":
    print("Testing Muon optimizer...")
    
    # Create test parameters
    weight_2d = torch.randn(512, 256, requires_grad=True)
    bias_1d = torch.randn(512, requires_grad=True)
    
    # Create optimizer
    optimizer = Muon([
        {'params': [weight_2d], 'lr': 0.02},
    ])
    
    # Simulate gradient
    weight_2d.grad = torch.randn_like(weight_2d)
    
    # Step
    optimizer.step()
    
    print("Muon step completed successfully!")
    
    # Test with backup
    optimizer_backup = MuonWithBackup([weight_2d, bias_1d], lr=0.02)
    weight_2d.grad = torch.randn_like(weight_2d)
    bias_1d.grad = torch.randn_like(bias_1d)
    
    optimizer_backup.step()
    print("MuonWithBackup step completed successfully!")

