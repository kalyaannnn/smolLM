# Quick Start Guide

## 🚀 Complete Setup in 3 Steps

### Step 1: Push to GitHub

```bash
cd /Users/kalyaanrao/finalVersion

# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit: SmolLM training pipeline"

# Push to GitHub
git remote add origin https://github.com/kalyaannnn/smolLM.git
git branch -M main
git push -u origin main
```

**Verify**: Visit https://github.com/kalyaannnn/smolLM

---

### Step 2: Open Colab & Run Setup

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `scripts/colab_setup.ipynb` OR create new notebook
3. Copy cells from the notebook
4. Run cells sequentially

**Or use this quick setup:**

```python
# Cell 1: Install
!pip install -q torch transformers datasets accelerate wandb pyyaml einops tqdm
!pip install -q flash-attn --no-build-isolation

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/smol-lm-checkpoints

# Cell 3: Clone
!git clone https://github.com/kalyaannnn/smolLM.git
%cd smolLM

# Cell 4: Sanity Check
!python sanity_check.py

# Cell 5: W&B Login (optional)
!wandb login

# Cell 6: Start Training
!python train_pretrain.py --config configs/pretrain.yaml
```

---

### Step 3: Monitor Training

- **W&B Dashboard**: Visit [wandb.ai](https://wandb.ai) → Project "smol-lm"
- **Checkpoints**: Saved to `/content/drive/MyDrive/smol-lm-checkpoints/`
- **Resume if disconnected**: `!python train_pretrain.py --config configs/pretrain.yaml --resume`

---

## ✅ Sanity Check Output

You should see:
```
✅ GPU: NVIDIA A100-SXM4-80GB
✅ VRAM: 80.0 GB
✅ Model initialized: 653.2M parameters
✅ Forward pass successful
✅ Backward pass successful
✅ Optimizer step successful
✅ All critical checks passed!
```

---

## 📊 What to Expect

- **Training time**: ~18 hours for 10B tokens
- **Checkpoints**: Every 500 steps (~2 hours)
- **W&B logs**: Every 10 steps
- **Validation**: Every 100 steps

---

## 🔧 Troubleshooting

### "No GPU detected"
- Make sure you're on **Colab Enterprise** (not regular Colab)
- Runtime → Change runtime type → GPU → A100

### "Flash Attention install failed"
- This is OK - code falls back to standard attention
- Training will be slightly slower but still works

### "W&B login failed"
- Optional - training still works with JSONL logs
- Get API key from: https://wandb.ai/authorize

---

## 📁 File Structure After Clone

```
smolLM/
├── configs/          # Training configs
├── models/           # Model architecture
├── data/             # Data pipeline
├── optim/            # Optimizers
├── train_pretrain.py # Main training script
├── sanity_check.py   # Verification script
└── scripts/          # Colab notebook
```

---

## 🎯 Next Steps After Pretraining

```bash
# SFT
!python train_sft.py --config configs/sft.yaml \
    --base-model /content/drive/MyDrive/smol-lm-checkpoints/pretrain_final.pt

# DPO
!python train_pref.py --config configs/dpo.yaml \
    --base-model /content/drive/MyDrive/smol-lm-checkpoints/sft_final.pt
```

---

**You're all set! 🚀**

