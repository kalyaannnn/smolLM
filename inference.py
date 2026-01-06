#!/usr/bin/env python3
"""
SmolLM Inference Script

Simple inference/demo script for text generation.

Usage:
    python inference.py --checkpoint path/to/checkpoint.pt --prompt "Hello, world"
    python inference.py --checkpoint path/to/checkpoint.pt --interactive
"""
import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.config import SmolLMConfig
from models.transformer import SmolLM
from data.tokenizer import load_tokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if "config" in checkpoint:
        if isinstance(checkpoint["config"], dict):
            config = SmolLMConfig.from_dict(checkpoint["config"].get("model", checkpoint["config"]))
        else:
            config = checkpoint["config"]
    else:
        print("Warning: No config in checkpoint, using defaults")
        config = SmolLMConfig.smol_600m()
    
    # Disable flash attention for inference simplicity
    config.use_flash_attn = False
    
    # Create model
    model = SmolLM(config)
    
    # Load weights
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def generate(
    model: SmolLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = "cuda",
) -> str:
    """Generate text from prompt."""
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer, device, args):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("SmolLM Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to clear context")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            if prompt.lower() == "clear":
                print("Context cleared.")
                continue
            
            if not prompt:
                continue
            
            # Generate response
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=not args.greedy,
                device=device,
            )
            
            # Print response (just the generated part)
            generated_part = response[len(prompt):].strip()
            print(f"SmolLM: {generated_part}\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="SmolLM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p filtering")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer name")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = load_tokenizer(args.tokenizer)
    
    print(f"\nModel loaded! Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Device: {args.device}")
    print(f"Precision: {next(model.parameters()).dtype}")
    
    if args.interactive:
        interactive_mode(model, tokenizer, args.device, args)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 40)
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=not args.greedy,
            device=args.device,
        )
        
        print(f"Generated:\n{response}")
    else:
        print("\nNo prompt provided. Use --prompt or --interactive")
        
        # Demo with a default prompt
        demo_prompt = "The capital of France is"
        print(f"\nDemo with prompt: '{demo_prompt}'")
        print("-" * 40)
        
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=demo_prompt,
            max_new_tokens=50,
            temperature=0.7,
            device=args.device,
        )
        
        print(f"Generated: {response}")


if __name__ == "__main__":
    main()

