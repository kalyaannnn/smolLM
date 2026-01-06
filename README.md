# SmolLM Training Pipeline

A complete, production-quality training pipeline for a ~600M parameter language model on a single A100 80GB GPU. Demonstrates end-to-end LLM development: **Pretrain → SFT → DPO → RLVR**.

## Features

### Architecture (~600M params)
- **GQA (Grouped Query Attention)**: n_kv_heads=4 for 3x KV cache reduction
- **Hybrid NoPE**: RoPE removed every 4th layer for better length generalization  
- **RMSNorm + SwiGLU**: Modern, efficient components
- **Tied embeddings**: Parameter efficient
- **Flash Attention 2**: 2-4x faster attention with memory efficiency

### Training
- **Muon + AdamW split**: Shape-based LR transfer for weight matrices
- **WSD scheduler**: Warmup-Stable-Decay for stable training
- **Deterministic data pipeline**: Exact reproducibility and resume
- **Sequence packing**: Document boundary masks prevent cross-document leakage

### Production Ready
- **Automatic checkpointing**: Google Drive support for Colab reliability
- **Cost tracking**: Tokens/dollar, ETA projections
- **Stability monitoring**: Loss spike and gradient anomaly detection
- **W&B logging**: With JSONL fallback
- **HuggingFace export**: Easy deployment

## Quick Start

### Installation

```bash
# Clone and setup
cd smol-lm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

### Training

```bash
# Pretraining (10B tokens)
python train_pretrain.py --config configs/pretrain.yaml

# Resume from checkpoint
python train_pretrain.py --config configs/pretrain.yaml --resume

# SFT (from pretrained checkpoint)
python train_sft.py --config configs/sft.yaml \
    --base-model /path/to/pretrain_final.pt

# DPO (from SFT checkpoint)
python train_pref.py --config configs/dpo.yaml \
    --base-model /path/to/sft_final.pt

# RLVR (optional, requires verifier)
python train_rlvr.py --config configs/rlvr.yaml \
    --base-model /path/to/dpo_final.pt \
    --verifier-model /path/to/verifier
```

### Inference

```bash
# Generate from checkpoint
python inference.py --checkpoint pretrain_final.pt \
    --prompt "The meaning of life is"

# Interactive mode
python inference.py --checkpoint pretrain_final.pt --interactive
```

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 1536 | Hidden dimension |
| n_layers | 24 | Transformer blocks |
| n_heads | 12 | Query heads |
| n_kv_heads | 4 | KV heads (GQA 3:1) |
| ffn_dim | 4096 | ~2.67x (SwiGLU parity) |
| vocab_size | 32000 | Llama 2 tokenizer |
| max_seq_len | 2048 | Context length |

**Total: ~580-620M parameters**

## Data Mix

| Domain | Weight | Sources |
|--------|--------|---------|
| Web | 87% | FineWeb-Edu (50%), DCLM (50%) |
| Code | 10% | The Stack |
| Math | 3% | OpenWebMath |

## Key Implementation Details

### Hybrid NoPE (RNoPE)
```python
# Layers 3, 7, 11, 15, 19, 23 skip RoPE
def is_nope_layer(layer_idx):
    return (layer_idx + 1) % 4 == 0
```

### Muon + AdamW Split
```python
# 2D weights → Muon with shape-based LR
# 1D params → AdamW
# Embeddings → AdamW without weight decay

lr_mult = sqrt(max(1, fan_out / fan_in))
# MLP up-projection (1536→6144) gets ~2x LR
```

### Document Masking
```python
# Packed sequences have block-diagonal attention
# Each document only attends to itself
mask[doc_start:doc_end, doc_start:doc_end] = True
```

### Truncated Normal Init
```python
std = 0.5 / sqrt(d_model)  # ≈ 0.0128
# Plus: multiply embedding output by sqrt(d_model)
```

## Project Structure

```
smol-lm/
├── configs/              # YAML configurations
│   ├── model/           # Model architecture configs
│   ├── pretrain.yaml    # Pretraining config
│   ├── sft.yaml         # SFT config
│   ├── dpo.yaml         # DPO config
│   └── rlvr.yaml        # RLVR config
├── data/                 # Data pipeline
│   ├── streaming.py     # Streaming dataloader
│   ├── packing.py       # Sequence packing + masks
│   └── tokenizer.py     # Tokenizer utilities
├── models/               # Model architecture
│   ├── config.py        # Model configuration
│   ├── layers.py        # RMSNorm, SwiGLU, GQA
│   └── transformer.py   # Full model
├── optim/                # Optimizers
│   ├── muon.py          # Muon optimizer
│   ├── param_groups.py  # Parameter grouping
│   └── scheduler.py     # WSD scheduler
├── train/                # Training utilities
│   └── stability.py     # Stability monitoring
├── utils/                # Utilities
│   ├── logging.py       # W&B + JSONL logging
│   ├── checkpoint.py    # Checkpoint management
│   └── cost_tracker.py  # Cost estimation
├── train_pretrain.py     # Pretraining script
├── train_sft.py          # SFT script
├── train_pref.py         # DPO/APO script
├── train_rlvr.py         # RLVR script
├── inference.py          # Inference script
└── requirements.txt
```

## Memory Requirements

For A100 80GB with BF16:

| Stage | Micro-batch | Grad Accum | Tokens/step | Memory |
|-------|-------------|------------|-------------|--------|
| Pretrain | 48 | 16 | 1.57M | ~25-30GB |
| SFT | 16 | 4 | 131K | ~15-20GB |
| DPO | 8 | 4 | 65K | ~25-30GB |

## Cost Estimates

For 10B token pretraining on A100 80GB (~$2.50/hr):
- ~150K tokens/second throughput
- ~18 hours training time
- ~$45 total cost

## Logging

Metrics logged to W&B (and local JSONL):
- `train/loss`, `train/grad_norm`, `train/lr`
- `val/loss`, `val/ppl` (per domain: web, code, math)
- `cost/usd_spent`, `cost/tokens_per_dollar`, `cost/eta_hours`
- `gpu/memory_allocated_gb`

## Checkpointing

Automatic checkpointing with:
- Save every N steps and/or every M hours
- Exact resume (optimizer + scheduler + RNG states)
- Google Drive support for Colab
- Checkpoint rotation (keep last N)

## References

- [SmolLM Blog Post](https://huggingface.co/blog/smollm)
- [Muon Optimizer (modded-nanogpt)](https://github.com/KellerJordan/modded-nanogpt)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [Grouped Query Attention](https://arxiv.org/abs/2305.13245)

## License

MIT

