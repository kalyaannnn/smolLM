# SmolLM Training Pipeline

A complete, production-quality training pipeline for a 160M-parameter language model on a single A100 80GB GPU. Demonstrates end-to-end LLM development: **Pretrain → SFT → DPO → RLVR**.

## Features

### Architecture
- **GQA (Grouped Query Attention)**: n_kv_heads=4 for 3x KV cache reduction
- **Hybrid NoPE**: RoPE removed every 4th layer for better length generalization  
- **RMSNorm + SwiGLU**: Modern, efficient components
- **Tied embeddings**: Parameter efficient
- **Flash Attention 2**: 2-4x faster attention with memory efficiency

**Model Option:**
- **160M**: d_model=768, n_layers=18, ~138M params - Fast training, good quality

### Training
- **AdamW optimizer**: Standard, stable training setup
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

### Step 1: Push to GitHub (if needed)

```bash
cd /Users/kalyaanrao/finalVersion
git init
git add .
git commit -m "Initial commit: SmolLM training pipeline"
git remote add origin https://github.com/kalyaannnn/smolLM.git
git branch -M main
git push -u origin main
```

### Step 2: Open Colab & Run Setup

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `scripts/colab_setup.ipynb` (or create a new notebook)
3. Run these cells in order:
   ```python
   # Install dependencies
   !pip install -q torch transformers datasets accelerate wandb pyyaml einops tqdm
   !pip install -q flash-attn --no-build-isolation
   
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/drive/MyDrive/smol-lm-checkpoints
   
   # Clone repo
   !git clone https://github.com/kalyaannnn/smolLM.git
   %cd smolLM
   import os, sys
   sys.path.append(os.getcwd())
   
   # Run sanity checks
   !python sanity_check.py
   
   # W&B login (optional)
   !wandb login
   
   # Start training (160M)
   !python train_pretrain.py --config configs/pretrain_160m.yaml
   ```

### Step 3: Monitor Training

- **W&B Dashboard**: visit [wandb.ai](https://wandb.ai) → project `smol-lm-160m`
- **Checkpoints**: `/content/drive/MyDrive/smol-lm-checkpoints/`
- **Resume**: `!python train_pretrain.py --config configs/pretrain_160m.yaml --resume`

### Local Setup

```bash
# Clone and setup
git clone https://github.com/kalyaannnn/smolLM.git
cd smolLM
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Flash Attention (requires CUDA)
pip install flash-attn --no-build-isolation

# Run sanity checks
python sanity_check.py
```

### Training

```bash
# Pretraining - 160M model (faster, lower compute)
python train_pretrain.py --config configs/pretrain_160m.yaml

# Resume from checkpoint
python train_pretrain.py --config configs/pretrain_160m.yaml --resume

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

## Model Configurations

### 160M Model (Fast Training)
| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 768 | Hidden dimension |
| n_layers | 18 | Transformer blocks |
| n_heads | 12 | Query heads |
| n_kv_heads | 4 | KV heads (GQA 3:1) |
| ffn_dim | 2048 | ~2.67x (SwiGLU parity) |
| vocab_size | 32000 | Llama 2 tokenizer |
| max_seq_len | 1024 | Context length |

**Total: ~138M parameters | Training time depends on token budget and throughput**


## Data Mix

| Domain | Weight | Sources |
|--------|--------|---------|
| Web | 87% | FineWeb-Edu (50%), DCLM (50%) |
| Code | 10% | CodeParrot |
| Math | 3% | OpenWebMath |

## Key Implementation Details

### Hybrid NoPE (RNoPE)
```python
# Layers 3, 7, 11, 15, 19, 23 skip RoPE
def is_nope_layer(layer_idx):
    return (layer_idx + 1) % 4 == 0
```

### Optimizer (AdamW)
```python
# AdamW with weight decay (no decay for embeddings/biases)
```

### Document Masking
```python
# Packed sequences use a block mask
# True = masked (disallowed) for sdpa
mask[doc_start:doc_end, doc_start:doc_end] = False
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
│   │   └── smol_160m.yaml
│   ├── pretrain_160m.yaml  # 160M pretraining config
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
│   ├── muon.py          # Optional Muon optimizer
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
├── sanity_check.py       # Pre-training verification
├── scripts/
│   └── colab_setup.ipynb # Colab setup notebook
├── requirements.txt
└── README.md
```

## Memory Requirements

Typical 160M (A100 80GB, BF16):

| Stage | Micro-batch | Grad Accum | Tokens/step |
|-------|-------------|------------|-------------|
| Pretrain | 32 | 48 | 1.57M |
| SFT | 16 | 4 | 65K |
| DPO | 8 | 4 | 32K |

## Cost & Time Estimates

Runtime scales with tokens/sec and token budget. Estimate:

```
time_seconds = total_tokens / tokens_per_second
```

## Logging & Monitoring

**W&B Integration**: Full metrics logging with JSONL fallback

**Training Metrics** (every 10 steps):
- `train/loss`, `train/grad_norm`, `train/lr`
- `train/tokens_per_sec`, `train/progress_percent`
- `gpu/memory_allocated_gb`

**Validation Metrics** (every 100 steps):
- `val/loss`, `val/ppl` (overall)
- `web_val_loss`, `web_val_ppl` (web domain)
- `code_val_loss`, `code_val_ppl` (code domain)
- `math_val_loss`, `math_val_ppl` (math domain)

**Cost Tracking**:
- `cost/usd_spent`, `cost/tokens_per_dollar`, `cost/eta_hours`

**Setup W&B**:
```bash
pip install wandb
wandb login  # Get API key from https://wandb.ai/authorize
```

## Checkpointing & Resume

Automatic checkpointing with:
- Save every N steps and/or every M hours
- Exact resume (optimizer + scheduler + RNG states)
- Google Drive support for Colab
- Checkpoint rotation (keep last N)

**Resume training**:
```bash
# Resume from latest checkpoint
python train_pretrain.py --config configs/pretrain_160m.yaml --resume

# Resume from specific checkpoint
python train_pretrain.py --config configs/pretrain_160m.yaml \
    --resume-from /path/to/checkpoint.pt
```

## Sanity Checks

Before training, verify everything works:
```bash
python sanity_check.py
```

Checks:
- ✅ GPU detection and VRAM
- ✅ Model initialization
- ✅ Forward/backward pass
- ✅ Optimizer setup
- ✅ Data loading
- ✅ W&B connection (optional)

## References

- [SmolLM Blog Post](https://huggingface.co/blog/smollm)
- [Muon Optimizer (optional reference)](https://github.com/KellerJordan/modded-nanogpt)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [Grouped Query Attention](https://arxiv.org/abs/2305.13245)

## Contributing

This is a demonstration project showing production-quality LLM training practices. Feel free to fork and adapt for your needs!

## License

MIT
