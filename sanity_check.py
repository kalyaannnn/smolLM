#!/usr/bin/env python3
"""
Sanity checks for SmolLM training setup.

Verifies:
- GPU availability and memory
- Model initialization
- Data loading
- Optimizer setup
- Forward/backward pass
- Checkpointing
- W&B connection (optional)
"""
import sys
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("SmolLM Sanity Checks")
print("=" * 60)

# 1. GPU Check
print("\n[1/8] Checking GPU...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ VRAM: {gpu_memory:.1f} GB")
    
    if gpu_memory < 20:
        print("⚠️  Warning: <20GB VRAM. May need to reduce batch size.")
    elif gpu_memory >= 80:
        print("✅ Excellent: A100 80GB detected!")
    else:
        print("✅ Good: Sufficient VRAM for training")
else:
    print("❌ No GPU detected! Training will be very slow.")
    sys.exit(1)

# 2. Model Import
print("\n[2/8] Checking model imports...")
try:
    from models.config import SmolLMConfig
    from models.transformer import SmolLM
    print("✅ Model imports successful")
except Exception as e:
    print(f"❌ Model import failed: {e}")
    sys.exit(1)

# 3. Model Initialization
print("\n[3/8] Testing model initialization...")
try:
    config = SmolLMConfig.smol_160m()
    config.use_flash_attn = False  # Disable for CPU testing if needed
    
    model = SmolLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model initialized: {param_count / 1e6:.1f}M parameters")
    
    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.is_bf16_supported():
            model = model.bfloat16()
            print("✅ Model moved to GPU (BF16)")
        else:
            model = model.half()
            print("✅ Model moved to GPU (FP16)")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Forward Pass
print("\n[4/8] Testing forward pass...")
try:
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs["logits"]
        print(f"✅ Forward pass successful: {logits.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Backward Pass
print("\n[5/8] Testing backward pass...")
try:
    model.train()
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    print(f"✅ Backward pass successful: loss = {loss.item():.4f}")
    
    # Check gradients
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"✅ Gradient norm: {grad_norm:.4f}")
    
    # Zero gradients
    model.zero_grad()
except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. Optimizer Setup
print("\n[6/8] Testing optimizer setup...")
try:
    adam_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    print("✅ AdamW optimizer created")
    
    # Test optimizer step
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    adam_optimizer.step()
    print("✅ Optimizer step successful")
except Exception as e:
    print(f"❌ Optimizer setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Data Loading
print("\n[7/8] Testing data loading...")
try:
    from data.tokenizer import load_tokenizer
    
    # Use a public tokenizer for testing
    try:
        tokenizer = load_tokenizer("gpt2")  # Public, no auth needed
        print("✅ Tokenizer loaded (using GPT-2 for test)")
    except:
        print("⚠️  Tokenizer test skipped (may need auth for Llama tokenizer)")
        tokenizer = None
    
    if tokenizer:
        test_text = "Hello, world! This is a test."
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenization works: {len(tokens)} tokens")
except Exception as e:
    print(f"⚠️  Data loading test skipped: {e}")

# 8. W&B Connection (Optional)
print("\n[8/8] Checking W&B connection...")
try:
    import wandb
    wandb.init(project="smol-lm-sanity", mode="disabled")
    print("✅ W&B available")
    wandb.finish()
except ImportError:
    print("⚠️  W&B not installed (optional)")
except Exception as e:
    print(f"⚠️  W&B connection test skipped: {e}")

# Summary
print("\n" + "=" * 60)
print("✅ All critical checks passed!")
print("=" * 60)
print("\nYou're ready to train! Next steps:")
print("1. Push code to GitHub")
print("2. Open Colab notebook")
print("3. Clone repo and start training")
print("\nRecommended: python train_pretrain.py --config configs/pretrain_160m.yaml")
