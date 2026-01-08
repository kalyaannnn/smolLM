"""
Core layers for SmolLM:
- RMSNorm (faster than LayerNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU MLP
- Grouped Query Attention (GQA) with optional Flash Attention
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Flash Attention (optional, requires CUDA)
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore[import-not-found]
    from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore[import-not-found]
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None  # type: ignore[assignment]
    flash_attn_varlen_func = None  # type: ignore[assignment]
    unpad_input = None  # type: ignore[assignment]
    pad_input = None  # type: ignore[assignment]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    ~30% faster than LayerNorm, no mean subtraction.
    
    Reference: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    
    Applies rotation to query and key vectors for position encoding.
    Reference: https://arxiv.org/abs/2104.09864
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # [seq_len, dim/2] -> [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to query and key tensors.
        
        Args:
            q: [batch, seq_len, n_heads, head_dim]
            k: [batch, seq_len, n_kv_heads, head_dim]
            position_ids: [batch, seq_len] optional position indices
            
        Returns:
            Rotated q, k tensors
        """
        seq_len = q.shape[1]
        
        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)
        
        if position_ids is None:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]
        
        # Apply rotation
        q_embed = self._apply_rotary(q, cos, sin)
        k_embed = self._apply_rotary(k, cos, sin)
        
        return q_embed, k_embed
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding to tensor."""
        # x: [batch, seq, heads, dim]
        # cos, sin: [seq, dim] or [batch, seq, dim]
        
        # Reshape for broadcasting
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(2)  # [batch, seq, 1, dim]
            sin = sin.unsqueeze(2)
        
        # Split into two halves and rotate
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([
            x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x1.shape[-1]],
            x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x1.shape[-1]],
        ], dim=-1)
        
        return rotated


class SwiGLU(nn.Module):
    """
    SwiGLU activation with gated linear unit.
    
    SwiGLU(x) = (x @ W_gate * SiLU(x @ W_up)) @ W_down
    
    Uses 3 weight matrices but is more expressive.
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * SiLU(up)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with optional Flash Attention.
    
    GQA uses fewer KV heads than query heads, reducing KV cache size.
    With n_heads=12 and n_kv_heads=4, each KV head serves 3 query heads.
    
    Reference: https://arxiv.org/abs/2305.13245
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        # Number of query heads per KV head
        self.n_rep = n_heads // n_kv_heads
        
        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        
        self.scale = self.head_dim ** -0.5
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads for GQA."""
        # x: [batch, seq, n_kv_heads, head_dim]
        if self.n_rep == 1:
            return x
        batch, seq_len, n_kv_heads, head_dim = x.shape
        x = x.unsqueeze(3).expand(batch, seq_len, n_kv_heads, self.n_rep, head_dim)
        return x.reshape(batch, seq_len, n_kv_heads * self.n_rep, head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        use_rope: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional RoPE and Flash Attention.
        
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: [batch, 1, seq_len, seq_len] or None for causal
            position_ids: [batch, seq_len] for RoPE
            rotary_emb: RotaryEmbedding module
            use_rope: Whether to apply RoPE (False for NoPE layers)
            cu_seqlens: Cumulative sequence lengths for Flash Attention varlen
            max_seqlen: Max sequence length for Flash Attention varlen
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE (skip for NoPE layers)
        if use_rope and rotary_emb is not None:
            q, k = rotary_emb(q, k, position_ids)
        
        # Use Flash Attention if available and no custom mask needed
        if self.use_flash_attn and attention_mask is None:
            # Flash attention expects [batch, seq, heads, dim]
            # Repeat KV for GQA
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
        elif self.use_flash_attn and cu_seqlens is not None:
            # Variable length attention for packed sequences
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
            
            # Reshape for varlen: [total_tokens, heads, dim]
            q = q.view(-1, self.n_heads, self.head_dim)
            k = k.view(-1, self.n_heads, self.head_dim)
            v = v.view(-1, self.n_heads, self.head_dim)
            
            output = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
            output = output.view(batch_size, seq_len, self.n_heads, self.head_dim)
        else:
            # Standard attention with optional mask
            output = self._standard_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_len, -1)
        return self.o_proj(output)
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention (fallback)."""
        # Repeat KV for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        
        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use PyTorch's efficient attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Create causal mask if none provided
            if attention_mask is None:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            # Manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is None:
                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(q.shape[2], k.shape[2], dtype=torch.bool, device=q.device),
                    diagonal=1
                )
                attn_weights.masked_fill_(causal_mask, float('-inf'))
            else:
                if attention_mask.dtype == torch.bool:
                    attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))
                else:
                    attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            output = torch.matmul(attn_weights, v)
        
        # Transpose back: [batch, seq, heads, dim]
        return output.transpose(1, 2)


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    - Pre-norm architecture
    - GQA attention (with optional RoPE/NoPE)
    - SwiGLU MLP
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        use_flash_attn: bool = True,
        layer_idx: int = 0,
        use_rope: bool = True,  # Set False for NoPE layers
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        
        # Pre-attention norm
        self.attn_norm = RMSNorm(d_model)
        
        # Attention
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=attention_dropout,
            use_flash_attn=use_flash_attn,
        )
        
        # Pre-MLP norm
        self.mlp_norm = RMSNorm(d_model)
        
        # MLP
        self.mlp = SwiGLU(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            rotary_emb: RoPE module
            cu_seqlens: For Flash Attention varlen
            max_seqlen: For Flash Attention varlen
        """
        # Attention with residual
        h = x + self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            rotary_emb=rotary_emb,
            use_rope=self.use_rope,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        
        # MLP with residual
        out = h + self.mlp(self.mlp_norm(h))
        
        return out
