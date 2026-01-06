"""
SmolLM Optimizers

- muon: Muon optimizer with Newton-Schulz orthogonalization
- param_groups: Smart parameter grouping (Muon vs AdamW split)
- scheduler: WSD (Warmup-Stable-Decay) scheduler
"""
from .muon import Muon, MuonWithBackup
from .param_groups import get_param_groups, create_optimizers
from .scheduler import WSDScheduler, WSDSchedulerDual

__all__ = [
    "Muon",
    "MuonWithBackup",
    "get_param_groups",
    "create_optimizers",
    "WSDScheduler",
    "WSDSchedulerDual",
]



