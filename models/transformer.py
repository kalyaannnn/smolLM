"""
SmolLM: ~600M parameter dense decoder-only Transformer.

Features:
- GQA with n_kv_heads=4
- Hybrid NoPE (RoPE removed every 4th layer)
- RMSNorm + SwiGLU
- Tied embeddings
- Truncated normal initialization
- Embedding output scaling
"""
import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SmolLMConfig
from .layers import (
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
)


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """
    Initialize tensor with truncated normal distribution.
    Values outside [mean - 2*std, mean + 2*std] are resampled.
    """
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
    return tensor


class SmolLM(nn.Module):
    """
    SmolLM: A small but capable language model.
    
    Architecture follows modern best practices:
    - Grouped Query Attention (GQA) for efficient KV cache
    - Hybrid NoPE for better length generalization
    - SwiGLU activation for expressivity
    - RMSNorm for faster normalization
    - Tied embeddings for parameter efficiency
    """
    
    def __init__(self, config: SmolLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Rotary embeddings (shared across layers that use RoPE)
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                use_flash_attn=config.use_flash_attn,
                layer_idx=i,
                use_rope=not config.is_nope_layer(i),  # Hybrid NoPE
            )
            for i in range(config.n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(config.d_model)
        
        # Output projection (tied with embeddings if configured)
        if config.tie_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Report param count
        self._report_params()
    
    def _init_weights(self):
        """
        Initialize weights with truncated normal distribution.
        std = init_std_factor / sqrt(d_model)
        """
        std = self.config.init_std
        
        # Initialize embeddings
        truncated_normal_(self.embed_tokens.weight, std=std)
        
        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                truncated_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize output projection if not tied
        if self.lm_head is not None:
            truncated_normal_(self.lm_head.weight, std=std)
    
    def _report_params(self):
        """Report parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Don't double-count tied embeddings
        if self.config.tie_embeddings:
            embed_params = self.embed_tokens.weight.numel()
            # Embedding is counted in total, no lm_head to add
        
        print(f"SmolLM initialized:")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Report NoPE layers
        nope_layers = [i for i in range(self.config.n_layers) if self.config.is_nope_layer(i)]
        if nope_layers:
            print(f"  NoPE layers (no RoPE): {nope_layers}")
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens
    
    def get_output_embeddings(self) -> nn.Module:
        if self.lm_head is not None:
            return self.lm_head
        return self.embed_tokens
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: [batch, 1, seq_len, seq_len] attention mask
                           For packed sequences, use cu_seqlens instead
            position_ids: [batch, seq_len] position indices (auto-generated if None)
            labels: [batch, seq_len] for language modeling loss
            cu_seqlens: [num_seqs + 1] cumulative sequence lengths for packed sequences
            max_seqlen: Maximum sequence length in batch (for Flash Attention varlen)
            return_hidden_states: Whether to return all hidden states
            
        Returns:
            Dictionary with:
            - logits: [batch, seq_len, vocab_size]
            - loss: scalar (if labels provided)
            - hidden_states: list of [batch, seq_len, d_model] (if requested)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get embeddings and scale
        hidden_states = self.embed_tokens(input_ids)
        if self.config.scale_embeddings:
            hidden_states = hidden_states * self.config.embedding_scale
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Store hidden states if requested
        all_hidden_states = [hidden_states] if return_hidden_states else None
        
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rotary_emb=self.rotary_emb,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if self.config.tie_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Build output
        output = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding/boundaries
            )
            output["loss"] = loss
        
        if return_hidden_states:
            output["hidden_states"] = all_hidden_states
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            do_sample: Whether to sample (False = greedy)
            eos_token_id: Stop token
            
        Returns:
            [batch, seq_len + new_tokens] generated sequence
        """
        self.eval()
        batch_size = input_ids.shape[0]
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # [batch, vocab]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append token
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    @classmethod
    def from_config(cls, config: SmolLMConfig) -> "SmolLM":
        """Create model from config."""
        return cls(config)
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "SmolLM":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if "config" in checkpoint:
            config = SmolLMConfig.from_dict(checkpoint["config"])
        else:
            config = SmolLMConfig()
        
        model = cls(config)
        
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model.to(device)
    
    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "config": self.config.to_dict(),
            "model_state": self.state_dict(),
        }
        torch.save(checkpoint, path)


# Quick test
if __name__ == "__main__":
    print("Testing SmolLM...")
    
    # Create config and model
    config = SmolLMConfig.smol_600m()
    
    # Disable flash attention for CPU testing
    config.use_flash_attn = False
    
    model = SmolLM(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    
    print(f"\nInput shape: {input_ids.shape}")
    
    outputs = model(input_ids, labels=labels)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    
    print("\nSmolLM test passed!")

