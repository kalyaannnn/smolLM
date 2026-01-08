"""
Validation and evaluation utilities.

Per-domain validation loss and perplexity tracking.
"""
import math
from typing import Dict, List, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.transformer import SmolLM
from data.tokenizer import load_tokenizer
from data.packing import PackingCollator


class ValidationEvaluator:
    """
    Evaluate model on held-out validation shards per domain.
    
    Computes:
    - Overall validation loss and perplexity
    - Per-domain loss and perplexity (web, code, math)
    """
    
    def __init__(
        self,
        model: SmolLM,
        tokenizer,
        config: Dict,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Load validation shards
        self.val_shards = self._load_validation_shards()
    
    def _load_validation_shards(self) -> Dict[str, List[List[int]]]:
        """Load fixed validation samples for each domain."""
        from datasets import load_dataset
        import random
        
        val_config = self.config.get("validation", {})
        shards = {}
        
        # Web validation
        if "web" in val_config:
            try:
                web_config = val_config["web"]
                ds = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    split="train",
                    streaming=True,
                )
                # Sample deterministically
                rng = random.Random(web_config.get("seed", 42))
                samples = []
                for i, item in enumerate(ds):
                    if i >= web_config.get("num_samples", 5000):
                        break
                    if i % 100 == 0:  # Sample every 100th for speed
                        text = item.get("text", "")
                        if text:
                            tokens = self.tokenizer.encode(
                                text,
                                add_special_tokens=False,
                                truncation=True,
                                max_length=2048,
                            )
                            if len(tokens) > 100:  # Filter very short docs
                                samples.append(tokens[:2048])  # Truncate
                shards["web"] = samples[:web_config.get("num_samples", 5000)]
            except Exception as e:
                print(f"Warning: Could not load web validation: {e}")
                shards["web"] = []
        
        # Code validation
        if "code" in val_config:
            try:
                code_config = val_config["code"]
                ds = load_dataset(
                    "codeparrot/codeparrot-clean",
                    split="train",
                    streaming=True,
                )
                rng = random.Random(code_config.get("seed", 42))
                samples = []
                for i, item in enumerate(ds):
                    if i >= code_config.get("num_samples", 2000):
                        break
                    if i % 50 == 0:
                        content = item.get("content", "")
                        if content:
                            tokens = self.tokenizer.encode(
                                content,
                                add_special_tokens=False,
                                truncation=True,
                                max_length=2048,
                            )
                            if len(tokens) > 100:
                                samples.append(tokens[:2048])
                shards["code"] = samples[:code_config.get("num_samples", 2000)]
            except Exception as e:
                print(f"Warning: Could not load code validation: {e}")
                shards["code"] = []
        
        # Math validation
        if "math" in val_config:
            try:
                math_config = val_config["math"]
                ds = load_dataset(
                    "open-web-math/open-web-math",
                    split="train",
                    streaming=True,
                )
                rng = random.Random(math_config.get("seed", 42))
                samples = []
                for i, item in enumerate(ds):
                    if i >= math_config.get("num_samples", 1000):
                        break
                    if i % 20 == 0:
                        text = item.get("text", "")
                        if text:
                            tokens = self.tokenizer.encode(
                                text,
                                add_special_tokens=False,
                                truncation=True,
                                max_length=2048,
                            )
                            if len(tokens) > 100:
                                samples.append(tokens[:2048])
                shards["math"] = samples[:math_config.get("num_samples", 1000)]
            except Exception as e:
                print(f"Warning: Could not load math validation: {e}")
                shards["math"] = []
        
        return shards
    
    @torch.no_grad()
    def evaluate(self, max_samples_per_domain: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model on validation shards.
        
        Returns:
            Dictionary with val_loss, val_ppl, and per-domain metrics
        """
        self.model.eval()
        
        results = {}
        total_loss = 0.0
        total_tokens = 0
        
        # Evaluate each domain
        for domain, shard in self.val_shards.items():
            if not shard:
                continue
            
            domain_samples = shard[:max_samples_per_domain] if max_samples_per_domain else shard
            
            domain_loss = 0.0
            domain_tokens = 0
            
            # Pack samples into batches
            collator = PackingCollator(
                seq_len=self.config["training"]["seq_len"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Process in small batches
            batch_size = 4
            for i in range(0, len(domain_samples), batch_size):
                batch_docs = domain_samples[i:i+batch_size]
                packed_batch = collator(batch_docs)
                
                input_ids = packed_batch.input_ids.to(self.device)
                labels = packed_batch.labels.to(self.device)
                attention_mask = packed_batch.attention_mask.to(self.device)
                position_ids = packed_batch.position_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                )
                
                loss = outputs["loss"]
                
                # Count non-ignored tokens
                num_tokens = (labels != -100).sum().item()
                
                domain_loss += loss.item() * num_tokens
                domain_tokens += num_tokens
            
            if domain_tokens > 0:
                domain_avg_loss = domain_loss / domain_tokens
                domain_ppl = math.exp(domain_avg_loss)
                
                results[f"{domain}_val_loss"] = domain_avg_loss
                results[f"{domain}_val_ppl"] = domain_ppl
                
                total_loss += domain_loss
                total_tokens += domain_tokens
        
        # Overall metrics
        if total_tokens > 0:
            overall_loss = total_loss / total_tokens
            overall_ppl = math.exp(overall_loss)
            
            results["val_loss"] = overall_loss
            results["val_ppl"] = overall_ppl
        
        self.model.train()
        return results


def run_validation(
    model: SmolLM,
    tokenizer,
    config: Dict,
    device: str = "cuda",
    max_samples_per_domain: Optional[int] = None,
) -> Dict[str, float]:
    """
    Quick validation run.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Training config
        device: Device
        max_samples_per_domain: Limit samples per domain (for speed)
        
    Returns:
        Validation metrics dictionary
    """
    evaluator = ValidationEvaluator(model, tokenizer, config, device)
    return evaluator.evaluate(max_samples_per_domain=max_samples_per_domain)
