"""
Smart parameter grouping for Muon + AdamW split optimizer.

Key insight: Different parameter types benefit from different optimizers.
- 2D weight matrices (attention, MLP): Muon with Newton-Schulz orthogonalization
- 1D params (biases, norms): AdamW
- Embeddings: AdamW without weight decay

Shape-based LR transfer:
- For rectangular matrices, adjust LR by sqrt(max(1, fan_out/fan_in))
- This means MLP up/gate projections (d_model -> ffn_dim) get ~2x LR
"""
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ParamGroupConfig:
    """Configuration for a parameter group."""
    name: str
    params: List[torch.nn.Parameter]
    lr: float
    weight_decay: float = 0.0
    optimizer_type: str = "adamw"  # "muon" or "adamw"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "name": self.name,
        }


def get_param_groups(
    model: nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    weight_decay: float = 0.1,
    verbose: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create parameter groups for Muon and AdamW optimizers.
    
    Strategy:
    - Muon: 2D weight matrices (Q, K, V, O, MLP up/gate/down)
    - AdamW: biases, norms, embeddings
    - Embeddings: AdamW but NO weight decay
    
    Shape-based LR for Muon:
    - lr_adjusted = muon_lr * sqrt(max(1, fan_out / fan_in))
    - MLP up/gate (d_model -> ffn_dim) get ~2x LR when ffn_dim = 4*d_model
    
    Args:
        model: The model to create param groups for
        muon_lr: Base learning rate for Muon (2D weights)
        adam_lr: Learning rate for AdamW (1D params)
        weight_decay: Weight decay for non-embedding params
        verbose: Print grouping statistics
        
    Returns:
        Tuple of (muon_param_groups, adam_param_groups)
    """
    muon_groups = []
    adam_params_with_wd = []
    adam_params_no_wd = []
    
    # Track stats
    muon_param_count = 0
    adam_wd_param_count = 0
    adam_no_wd_param_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Categorize parameters
        if param.ndim == 2:
            # 2D weight matrices -> Muon with shape-based LR
            fan_out, fan_in = param.shape
            lr_mult = math.sqrt(max(1, fan_out / fan_in))
            adjusted_lr = muon_lr * lr_mult
            
            muon_groups.append({
                "params": [param],
                "lr": adjusted_lr,
                "name": name,
                "shape": list(param.shape),
                "lr_mult": lr_mult,
            })
            muon_param_count += param.numel()
            
        elif "embed" in name.lower() or "token" in name.lower():
            # Embeddings -> AdamW, no weight decay
            adam_params_no_wd.append(param)
            adam_no_wd_param_count += param.numel()
            
        else:
            # Biases, norms, etc -> AdamW with weight decay
            adam_params_with_wd.append(param)
            adam_wd_param_count += param.numel()
    
    # Create AdamW groups
    adam_groups = []
    if adam_params_with_wd:
        adam_groups.append({
            "params": adam_params_with_wd,
            "lr": adam_lr,
            "weight_decay": weight_decay,
            "name": "adam_with_wd",
        })
    if adam_params_no_wd:
        adam_groups.append({
            "params": adam_params_no_wd,
            "lr": adam_lr,
            "weight_decay": 0.0,
            "name": "adam_no_wd (embeddings)",
        })
    
    if verbose:
        total = muon_param_count + adam_wd_param_count + adam_no_wd_param_count
        print(f"\nParameter grouping for Muon + AdamW:")
        print(f"  Muon (2D weights): {muon_param_count:,} params ({100*muon_param_count/total:.1f}%)")
        print(f"  AdamW (with WD):   {adam_wd_param_count:,} params ({100*adam_wd_param_count/total:.1f}%)")
        print(f"  AdamW (no WD):     {adam_no_wd_param_count:,} params ({100*adam_no_wd_param_count/total:.1f}%)")
        print(f"  Total:             {total:,} params")
        
        # Show some Muon LR adjustments
        print(f"\nMuon LR adjustments (shape-based):")
        for group in muon_groups[:5]:  # Show first 5
            print(f"  {group['name']}: {group['shape']} -> lr_mult={group['lr_mult']:.2f}, lr={group['lr']:.4f}")
        if len(muon_groups) > 5:
            print(f"  ... and {len(muon_groups) - 5} more groups")
    
    return muon_groups, adam_groups


def create_optimizers(
    model: nn.Module,
    muon_lr: float = 0.02,
    adam_lr: float = 3e-4,
    weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-8,
    verbose: bool = True,
) -> Tuple["Muon", "torch.optim.AdamW"]:
    """
    Create Muon and AdamW optimizers with proper parameter grouping.
    
    Args:
        model: Model to optimize
        muon_lr: Base LR for Muon (applied to 2D weights)
        adam_lr: LR for AdamW (1D params and embeddings)
        weight_decay: Weight decay (not applied to embeddings)
        muon_momentum: Momentum for Muon
        adam_betas: Betas for AdamW
        adam_eps: Epsilon for AdamW
        verbose: Print grouping info
        
    Returns:
        Tuple of (muon_optimizer, adam_optimizer)
    """
    from .muon import Muon
    
    muon_groups, adam_groups = get_param_groups(
        model,
        muon_lr=muon_lr,
        adam_lr=adam_lr,
        weight_decay=weight_decay,
        verbose=verbose,
    )
    
    # Create optimizers
    muon_optimizer = Muon(
        muon_groups,
        lr=muon_lr,  # Base LR (overridden per group)
        momentum=muon_momentum,
    )
    
    adam_optimizer = torch.optim.AdamW(
        adam_groups,
        lr=adam_lr,  # Base LR (overridden per group)
        betas=adam_betas,
        eps=adam_eps,
    )
    
    return muon_optimizer, adam_optimizer


def get_num_params_per_optimizer(
    model: nn.Module,
) -> Dict[str, int]:
    """Get parameter count breakdown by optimizer type."""
    muon_count = 0
    adam_wd_count = 0
    adam_no_wd_count = 0
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim == 2:
            muon_count += param.numel()
        elif "embed" in name.lower() or "token" in name.lower():
            adam_no_wd_count += param.numel()
        else:
            adam_wd_count += param.numel()
    
    return {
        "muon": muon_count,
        "adam_with_wd": adam_wd_count,
        "adam_no_wd": adam_no_wd_count,
        "total": muon_count + adam_wd_count + adam_no_wd_count,
    }


# Quick test
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models.config import SmolLMConfig
    from models.transformer import SmolLM
    
    print("Testing parameter grouping...")
    
    # Create model
    config = SmolLMConfig.smol_600m()
    config.use_flash_attn = False  # For CPU testing
    model = SmolLM(config)
    
    # Get param groups
    muon_groups, adam_groups = get_param_groups(
        model,
        muon_lr=0.02,
        adam_lr=3e-4,
        weight_decay=0.1,
        verbose=True,
    )
    
    print(f"\nCreated {len(muon_groups)} Muon groups and {len(adam_groups)} AdamW groups")
    
    # Get stats
    stats = get_num_params_per_optimizer(model)
    print(f"\nParameter counts: {stats}")

