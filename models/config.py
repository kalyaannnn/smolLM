"""
Model configuration for SmolLM.
~600M parameter dense decoder-only Transformer.
"""
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class SmolLMConfig:
    """
    Configuration for SmolLM ~600M model.
    
    Architecture:
    - GQA with n_kv_heads=4
    - RMSNorm + SwiGLU
    - Hybrid NoPE (RoPE removed every 4th layer)
    - Tied embeddings
    """
    # Model dimensions
    d_model: int = 1536
    n_layers: int = 24
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA: 3:1 ratio for memory efficiency
    
    # FFN dimensions (SwiGLU uses 3 matrices)
    # For ~600M params with SwiGLU: ffn_dim ≈ 4 * d_model * 2/3
    ffn_dim: int = 4096
    
    # Vocabulary (matches Llama 2 tokenizer)
    vocab_size: int = 32000
    
    # Sequence length
    max_seq_len: int = 2048
    
    # RoPE parameters
    rope_theta: float = 10000.0
    
    # Hybrid NoPE: remove RoPE every N layers (0 = always use RoPE)
    # With nope_layer_interval=4, layers 3,7,11,15,19,23 have no RoPE
    nope_layer_interval: int = 4
    
    # Dropout (usually 0 for pretraining)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    
    # Initialization
    # Truncated normal: std = init_std_factor / sqrt(d_model)
    init_std_factor: float = 0.5
    
    # Scale embedding output by sqrt(d_model)
    scale_embeddings: bool = True
    
    # Tie input/output embeddings
    tie_embeddings: bool = True
    
    # Flash Attention
    use_flash_attn: bool = True
    
    # Precision
    dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    
    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_heads
    
    @property
    def kv_dim(self) -> int:
        """Total KV dimension (for GQA)."""
        return self.n_kv_heads * self.head_dim
    
    @property
    def init_std(self) -> float:
        """Standard deviation for truncated normal init."""
        return self.init_std_factor / math.sqrt(self.d_model)
    
    @property
    def embedding_scale(self) -> float:
        """Scale factor for embedding output."""
        return math.sqrt(self.d_model) if self.scale_embeddings else 1.0
    
    def is_nope_layer(self, layer_idx: int) -> bool:
        """Check if a layer should skip RoPE (hybrid NoPE)."""
        if self.nope_layer_interval == 0:
            return False
        # Every 4th layer starting from layer 3 (0-indexed)
        return (layer_idx + 1) % self.nope_layer_interval == 0
    
    def num_params(self, include_embeddings: bool = True) -> int:
        """Estimate total parameter count."""
        # Embeddings
        embed_params = self.vocab_size * self.d_model
        if self.tie_embeddings:
            embed_params = embed_params  # counted once
        else:
            embed_params = embed_params * 2  # input + output
        
        # Per-layer params
        # Attention: Q + K + V + O projections
        # Q: d_model -> d_model
        # K, V: d_model -> kv_dim (GQA)
        # O: d_model -> d_model
        attn_params = (
            self.d_model * self.d_model +  # Q
            self.d_model * self.kv_dim +    # K
            self.d_model * self.kv_dim +    # V
            self.d_model * self.d_model     # O
        )
        
        # MLP: SwiGLU has gate, up, down projections
        # gate: d_model -> ffn_dim
        # up: d_model -> ffn_dim
        # down: ffn_dim -> d_model
        mlp_params = 3 * self.d_model * self.ffn_dim
        
        # RMSNorm: 2 per layer (pre-attn, pre-mlp)
        norm_params = 2 * self.d_model
        
        per_layer = attn_params + mlp_params + norm_params
        total_layers = per_layer * self.n_layers
        
        # Final norm
        final_norm = self.d_model
        
        if include_embeddings:
            return embed_params + total_layers + final_norm
        return total_layers + final_norm
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
    
    @classmethod
    def smol_600m(cls) -> "SmolLMConfig":
        """Default 600M parameter configuration."""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SmolLMConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "ffn_dim": self.ffn_dim,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "nope_layer_interval": self.nope_layer_interval,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "init_std_factor": self.init_std_factor,
            "scale_embeddings": self.scale_embeddings,
            "tie_embeddings": self.tie_embeddings,
            "use_flash_attn": self.use_flash_attn,
            "dtype": self.dtype,
        }


# Quick test
if __name__ == "__main__":
    config = SmolLMConfig.smol_600m()
    print(f"SmolLM Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads} (head_dim: {config.head_dim})")
    print(f"  n_kv_heads: {config.n_kv_heads} (kv_dim: {config.kv_dim})")
    print(f"  ffn_dim: {config.ffn_dim}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  Estimated params: {config.num_params() / 1e6:.1f}M")
    print(f"  Init std: {config.init_std:.6f}")
    print(f"  Embedding scale: {config.embedding_scale:.2f}")
    print(f"\nHybrid NoPE layers (no RoPE):")
    for i in range(config.n_layers):
        if config.is_nope_layer(i):
            print(f"  Layer {i}")

