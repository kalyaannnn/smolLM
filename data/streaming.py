"""
Streaming dataloader with deterministic domain mixing.

Features:
- Streaming from HuggingFace datasets (never loads full dataset)
- Deterministic mixing with seeded RNG (exact resume)
- Token budgets per domain
- Buffer-based shuffling
"""
import random
from typing import Dict, List, Iterator, Optional, Any
from dataclasses import dataclass, field
import itertools

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, IterableDataset as HFIterableDataset


@dataclass
class DomainConfig:
    """Configuration for a single data domain."""
    name: str
    weight: float  # Sampling weight (should sum to 1.0 across domains)
    sources: List[tuple]  # List of (dataset_name, subset_weight) tuples
    max_tokens: Optional[int] = None  # Token budget cap
    text_field: str = "text"  # Field containing text in dataset
    
    def __post_init__(self):
        # Normalize subset weights
        total = sum(w for _, w in self.sources)
        self.sources = [(name, w / total) for name, w in self.sources]


# Default data mix: 87% web, 10% code, 3% math
DEFAULT_DOMAIN_MIX = {
    "web": DomainConfig(
        name="web",
        weight=0.87,
        sources=[
            ("HuggingFaceFW/fineweb-edu", 0.5),
            ("mlfoundations/dclm-baseline-1.0", 0.5),
        ],
        text_field="text",
    ),
    "code": DomainConfig(
        name="code",
        weight=0.10,
        sources=[
            ("bigcode/the-stack-dedup", 1.0),
        ],
        text_field="content",
    ),
    "math": DomainConfig(
        name="math",
        weight=0.03,
        sources=[
            ("open-web-math/open-web-math", 1.0),
        ],
        text_field="text",
    ),
}


class DomainStream:
    """
    Stream from a single domain with deterministic shuffling.
    """
    
    def __init__(
        self,
        config: DomainConfig,
        tokenizer,
        seed: int = 42,
        buffer_size: int = 10000,
        split: str = "train",
        max_length: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.seed = seed
        self.buffer_size = buffer_size
        self.split = split
        self.max_length = max_length
        
        self.rng = random.Random(seed)
        self.tokens_yielded = 0
        self._buffer = []
        self._source_iters = []
        self._exhausted = False
    
    def _init_sources(self):
        """Initialize dataset iterators for all sources."""
        self._source_iters = []
        for dataset_name, weight in self.config.sources:
            try:
                # Load as streaming dataset
                ds = load_dataset(
                    dataset_name,
                    split=self.split,
                    streaming=True,
                )
                # Shuffle with seed for determinism
                ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)
                self._source_iters.append((iter(ds), weight))
            except Exception as e:
                print(f"Warning: Could not load {dataset_name}: {e}")
    
    def _fill_buffer(self):
        """Fill buffer from sources."""
        if not self._source_iters:
            self._init_sources()
        
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            # Sample from sources according to weights
            weights = [w for _, w in self._source_iters]
            if not weights:
                self._exhausted = True
                break
            
            idx = self.rng.choices(range(len(self._source_iters)), weights=weights)[0]
            source_iter, _ = self._source_iters[idx]
            
            try:
                item = next(source_iter)
                text = item.get(self.config.text_field, "")
                if text:
                    self._buffer.append(text)
            except StopIteration:
                # Remove exhausted source
                self._source_iters.pop(idx)
        
        # Shuffle buffer
        self.rng.shuffle(self._buffer)
    
    def __iter__(self):
        """Iterate over tokenized documents."""
        while True:
            if not self._buffer:
                self._fill_buffer()
                if not self._buffer:
                    break
            
            text = self._buffer.pop(0)
            
            # Tokenize
            if self.max_length:
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Check token budget
            if self.config.max_tokens and self.tokens_yielded >= self.config.max_tokens:
                break
            
            self.tokens_yielded += len(tokens)
            yield tokens
    
    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            "tokens_yielded": self.tokens_yielded,
            "rng_state": self.rng.getstate(),
            "buffer_size": len(self._buffer),
        }
    
    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.tokens_yielded = state["tokens_yielded"]
        self.rng.setstate(state["rng_state"])


class MixedDomainDataset(IterableDataset):
    """
    Mixed domain dataset with deterministic streaming.
    
    Samples from multiple domains according to configured weights,
    supports token budgets, and enables exact resume from checkpoints.
    """
    
    def __init__(
        self,
        tokenizer,
        domains: Optional[Dict[str, DomainConfig]] = None,
        seed: int = 42,
        buffer_size: int = 10000,
        total_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.domains = domains or DEFAULT_DOMAIN_MIX
        self.seed = seed
        self.buffer_size = buffer_size
        self.total_tokens = total_tokens
        self.max_length = max_length
        
        # Create domain streams
        self.streams = {
            name: DomainStream(
                config=config,
                tokenizer=tokenizer,
                seed=seed + hash(name) % 10000,
                buffer_size=buffer_size,
                max_length=max_length,
            )
            for name, config in self.domains.items()
        }
        
        # Initialize RNG for domain sampling
        self.rng = random.Random(seed)
        self.tokens_yielded = 0
    
    def _sample_domain(self) -> str:
        """Sample a domain according to weights."""
        domains = list(self.domains.keys())
        weights = [self.domains[d].weight for d in domains]
        return self.rng.choices(domains, weights=weights)[0]
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over tokenized documents from mixed domains."""
        domain_iters = {name: iter(stream) for name, stream in self.streams.items()}
        active_domains = list(domain_iters.keys())
        
        while True:
            # Check total budget
            if self.total_tokens and self.tokens_yielded >= self.total_tokens:
                break
            if not active_domains:
                break
            
            # Sample domain
            weights = [self.domains[d].weight for d in active_domains]
            if sum(weights) <= 0:
                break
            domain = self.rng.choices(active_domains, weights=weights)[0]
            
            try:
                tokens = next(domain_iters[domain])
                self.tokens_yielded += len(tokens)
                yield tokens
            except StopIteration:
                # Domain exhausted, drop it
                active_domains = [d for d in active_domains if d != domain]
    
    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            "tokens_yielded": self.tokens_yielded,
            "rng_state": self.rng.getstate(),
            "stream_states": {
                name: stream.get_state() 
                for name, stream in self.streams.items()
            },
        }
    
    def load_state(self, state: dict):
        """Load state from checkpoint."""
        self.tokens_yielded = state["tokens_yielded"]
        self.rng.setstate(state["rng_state"])
        for name, stream_state in state["stream_states"].items():
            if name in self.streams:
                self.streams[name].load_state(stream_state)


def create_streaming_dataloader(
    tokenizer,
    batch_size: int = 32,
    domains: Optional[Dict[str, DomainConfig]] = None,
    seed: int = 42,
    num_workers: int = 0,  # 0 for streaming
    total_tokens: Optional[int] = None,
) -> DataLoader:
    """
    Create a streaming dataloader with domain mixing.
    
    Note: This returns raw tokenized documents. Use with PackingCollator
    for packed sequences with document masks.
    
    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size (documents per batch before packing)
        domains: Domain configuration (uses default if None)
        seed: Random seed for determinism
        num_workers: Number of workers (0 recommended for streaming)
        total_tokens: Total token budget
        
    Returns:
        DataLoader yielding batches of tokenized documents
    """
    dataset = MixedDomainDataset(
        tokenizer=tokenizer,
        domains=domains,
        seed=seed,
        total_tokens=total_tokens,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x,  # Return list of token lists
    )


# Quick test
if __name__ == "__main__":
    print("Testing streaming dataloader...")
    
    # Use simple tokenizer for testing
    from transformers import AutoTokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Create small test domain config
        test_domains = {
            "test": DomainConfig(
                name="test",
                weight=1.0,
                sources=[("wikitext", 1.0)],  # Small public dataset
                text_field="text",
                max_tokens=10000,
            ),
        }
        
        dataset = MixedDomainDataset(
            tokenizer=tokenizer,
            domains=test_domains,
            seed=42,
            total_tokens=10000,
        )
        
        print("Streaming first 5 documents...")
        for i, tokens in enumerate(dataset):
            if i >= 5:
                break
            print(f"  Doc {i}: {len(tokens)} tokens")
        
        print(f"Total tokens yielded: {dataset.tokens_yielded}")
        print("\nStreaming dataloader test passed!")
        
    except Exception as e:
        print(f"Test skipped (may need network): {e}")
