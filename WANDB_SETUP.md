# W&B (Weights & Biases) Setup & Monitoring

## ✅ Already Configured!

W&B is fully integrated into the training pipeline. Here's what you get:

---

## What Gets Logged to W&B

### Training Metrics (Every 10 steps)
- `train/loss` - Training loss
- `train/grad_norm` - Gradient norm (for stability)
- `train/lr` - AdamW learning rate
- `train/tokens_per_sec` - Training throughput
- `train/tokens_processed` - Total tokens seen
- `train/progress_percent` - Training progress

### Cost Tracking
- `cost/usd_spent` - Estimated cost so far
- `cost/tokens_per_dollar` - Training efficiency
- `cost/eta_hours` - Estimated time to completion

### GPU Metrics
- `gpu/memory_allocated_gb` - GPU memory used
- `gpu/memory_reserved_gb` - GPU memory reserved

### Validation Metrics (Every 100 steps)
- `val_loss` - Overall validation loss
- `val_ppl` - Overall validation perplexity
- `web_val_loss` / `web_val_ppl` - Web domain metrics
- `code_val_loss` / `code_val_ppl` - Code domain metrics
- `math_val_loss` / `math_val_ppl` - Math domain metrics

### Stability Alerts
- Automatic alerts for loss spikes, gradient explosions, NaN values

---

## Setup (One-Time)

### 1. Install W&B
```bash
pip install wandb
```

### 2. Login
```bash
wandb login
# Enter your API key from: https://wandb.ai/authorize
```

### 3. That's it! Training will auto-log to W&B

---

## Viewing Your Runs

1. Go to [wandb.ai](https://wandb.ai)
2. Find your project: **"smol-lm"**
3. See all runs, compare experiments, view metrics

---

## Configuration

In `configs/pretrain.yaml`:
```yaml
logging:
  project: "smol-lm"        # W&B project name
  use_wandb: true           # Enable/disable W&B
  log_every_steps: 10       # How often to log
  eval_every_steps: 100     # How often to run validation
```

---

## Fallback: JSONL Logs

Even if W&B fails, all metrics are saved to:
```
./logs/{run_name}.jsonl
```

You can always parse these later!

---

## Example W&B Dashboard

You'll see:
- **Loss curves** (train + validation)
- **Per-domain perplexity** (web/code/math)
- **Learning rate schedules**
- **GPU utilization**
- **Cost tracking**
- **Training stability** (gradient norms, alerts)

---

## Quick Test

```python
# Test W&B connection
import wandb
wandb.init(project="smol-lm-test", mode="online")
wandb.log({"test": 1.0})
wandb.finish()
```

If this works, you're ready to train! 🚀
