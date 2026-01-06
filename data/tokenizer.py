"""
Tokenizer utilities - wrapper for pre-trained tokenizers.

We use existing tokenizers (no training needed):
- Llama 2: 32K vocab, good general purpose
- Mistral: 32K vocab, modern
- GPT-NeoX: 50K vocab, good code support
"""
from typing import Optional, List, Union
from transformers import AutoTokenizer, PreTrainedTokenizer


# Recommended tokenizers
TOKENIZER_PRESETS = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "mistral": "mistralai/Mistral-7B-v0.1",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
    "llama3": "meta-llama/Meta-Llama-3-8B",
}


def load_tokenizer(
    name_or_path: str = "meta-llama/Llama-2-7b-hf",
    use_fast: bool = True,
    add_bos: bool = True,
    add_eos: bool = True,
) -> PreTrainedTokenizer:
    """
    Load a pre-trained tokenizer from HuggingFace.
    
    No tokenizer training needed - we reuse existing tokenizers.
    
    Args:
        name_or_path: HuggingFace model name or local path
                     Can also use preset names: "llama2", "mistral", "gpt-neox"
        use_fast: Use fast tokenizer (recommended)
        add_bos: Add BOS token by default
        add_eos: Add EOS token by default
        
    Returns:
        Configured tokenizer
    """
    # Handle presets
    if name_or_path in TOKENIZER_PRESETS:
        name_or_path = TOKENIZER_PRESETS[name_or_path]
    
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=use_fast,
        trust_remote_code=True,
    )
    
    # Ensure special tokens exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set default behavior
    tokenizer.add_bos_token = add_bos
    tokenizer.add_eos_token = add_eos
    
    return tokenizer


def get_vocab_size(tokenizer: PreTrainedTokenizer) -> int:
    """Get vocabulary size from tokenizer."""
    return len(tokenizer)


def tokenize_text(
    tokenizer: PreTrainedTokenizer,
    text: Union[str, List[str]],
    max_length: Optional[int] = None,
    truncation: bool = True,
    padding: bool = False,
    return_tensors: Optional[str] = None,
) -> dict:
    """
    Tokenize text with standard settings.
    
    Args:
        tokenizer: The tokenizer to use
        text: Single string or list of strings
        max_length: Maximum sequence length
        truncation: Whether to truncate
        padding: Whether to pad
        return_tensors: "pt" for PyTorch, "np" for NumPy, None for lists
        
    Returns:
        Dictionary with input_ids, attention_mask, etc.
    """
    return tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors,
    )


def decode_tokens(
    tokenizer: PreTrainedTokenizer,
    token_ids: Union[List[int], "torch.Tensor"],
    skip_special_tokens: bool = True,
) -> str:
    """Decode token IDs back to text."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class TokenizerWrapper:
    """
    Convenience wrapper for tokenizer with common operations.
    """
    
    def __init__(
        self,
        name_or_path: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 2048,
    ):
        self.tokenizer = load_tokenizer(name_or_path)
        self.max_length = max_length
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """Encode text to token IDs."""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
    
    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return decode_tokens(self.tokenizer, token_ids, skip_special_tokens)
    
    def __call__(self, text, **kwargs):
        """Tokenize text."""
        kwargs.setdefault("max_length", self.max_length)
        kwargs.setdefault("truncation", True)
        return self.tokenizer(text, **kwargs)


# Quick test
if __name__ == "__main__":
    print("Testing tokenizer utilities...")
    
    # Note: This requires HuggingFace login for Llama tokenizers
    # For testing without auth, use a public tokenizer
    try:
        tokenizer = load_tokenizer("gpt2")  # Public, no auth needed
        print(f"Loaded tokenizer with vocab size: {get_vocab_size(tokenizer)}")
        
        # Test tokenization
        text = "Hello, world! This is a test of the tokenizer."
        tokens = tokenize_text(tokenizer, text)
        print(f"Tokenized: {tokens['input_ids'][:10]}...")
        
        # Test decode
        decoded = decode_tokens(tokenizer, tokens['input_ids'])
        print(f"Decoded: {decoded}")
        
        print("\nTokenizer test passed!")
    except Exception as e:
        print(f"Tokenizer test skipped (may need auth): {e}")



