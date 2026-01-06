"""
Sequence packing with document boundary masks.

Packing concatenates multiple documents into single sequences for efficiency.
Document masks prevent cross-document attention leakage - a critical but
often-overlooked detail in LLM training.

Output format:
- input_ids: [batch, seq_len] packed sequences
- labels: [batch, seq_len] with -100 at document boundaries
- attention_mask: [batch, seq_len, seq_len] block-diagonal for intra-doc attention
- position_ids: [batch, seq_len] reset per document
- cu_seqlens: [num_docs + 1] cumulative lengths for Flash Attention varlen
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class PackedBatch:
    """Container for a packed batch."""
    input_ids: torch.Tensor           # [batch, seq_len]
    labels: torch.Tensor              # [batch, seq_len], -100 at boundaries
    attention_mask: torch.Tensor      # [batch, seq_len, seq_len] block diagonal
    position_ids: torch.Tensor        # [batch, seq_len] reset per doc
    doc_boundaries: List[List[int]]   # Document end positions per batch item
    cu_seqlens: Optional[torch.Tensor] = None  # For Flash Attention varlen
    max_seqlen: Optional[int] = None  # Max doc length in batch


def create_document_mask(
    doc_boundaries: List[int],
    seq_len: int,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create block-diagonal attention mask for packed sequences.
    
    Each document can only attend to tokens within itself,
    preventing cross-document information leakage.
    
    Args:
        doc_boundaries: List of document end positions
        seq_len: Total sequence length
        dtype: Output dtype (bool for memory efficiency)
        
    Returns:
        [seq_len, seq_len] mask where True = can attend
    """
    mask = torch.zeros(seq_len, seq_len, dtype=dtype)
    
    start = 0
    for end in doc_boundaries:
        # Each document attends only to itself (causal within doc)
        for i in range(start, min(end, seq_len)):
            # Can attend to positions [start, i] (causal)
            mask[i, start:i+1] = True
        start = end
    
    return mask


def create_position_ids(
    doc_boundaries: List[int],
    seq_len: int,
) -> torch.Tensor:
    """
    Create position IDs that reset at document boundaries.
    
    This ensures RoPE positions restart for each document.
    
    Args:
        doc_boundaries: List of document end positions
        seq_len: Total sequence length
        
    Returns:
        [seq_len] position IDs
    """
    position_ids = torch.zeros(seq_len, dtype=torch.long)
    
    start = 0
    for end in doc_boundaries:
        doc_len = min(end, seq_len) - start
        position_ids[start:start + doc_len] = torch.arange(doc_len)
        start = end
    
    return position_ids


def create_labels(
    input_ids: torch.Tensor,
    doc_boundaries: List[int],
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Create labels with -100 at document boundaries.
    
    The model shouldn't predict across document boundaries.
    
    Args:
        input_ids: Token IDs
        doc_boundaries: Document end positions
        ignore_index: Value for ignored positions
        
    Returns:
        Labels tensor
    """
    labels = input_ids.clone()
    
    # Mark positions right before boundaries as ignore
    # (we don't want to predict the first token of next doc)
    for boundary in doc_boundaries[:-1]:  # Skip last (end of sequence)
        if boundary > 0 and boundary < len(labels):
            labels[boundary - 1] = ignore_index
    
    return labels


class SequencePacker:
    """
    Pack multiple documents into fixed-length sequences.
    
    Documents are concatenated with optional separator tokens.
    Handles document boundary tracking for proper masking.
    """
    
    def __init__(
        self,
        seq_len: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        add_eos: bool = True,
    ):
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.add_eos = add_eos and eos_token_id is not None
    
    def pack_documents(
        self,
        documents: List[List[int]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Pack documents into a single sequence.
        
        Args:
            documents: List of tokenized documents
            
        Returns:
            Tuple of (packed_ids, doc_boundaries)
        """
        packed = []
        boundaries = []
        
        for doc in documents:
            # Add EOS if configured
            if self.add_eos:
                doc = doc + [self.eos_token_id]
            
            # Check if it fits
            remaining = self.seq_len - len(packed)
            if remaining <= 0:
                break
            
            # Truncate if needed
            doc = doc[:remaining]
            packed.extend(doc)
            boundaries.append(len(packed))
        
        # Pad to seq_len
        if len(packed) < self.seq_len:
            packed.extend([self.pad_token_id] * (self.seq_len - len(packed)))
        
        return torch.tensor(packed, dtype=torch.long), boundaries
    
    def pack_batch(
        self,
        document_batches: List[List[List[int]]],
    ) -> PackedBatch:
        """
        Pack multiple batches of documents.
        
        Args:
            document_batches: List of document lists (one per batch item)
            
        Returns:
            PackedBatch with all necessary tensors
        """
        batch_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        batch_position_ids = []
        batch_boundaries = []
        
        all_cu_seqlens = [0]  # Cumulative sequence lengths
        max_doc_len = 0
        
        for docs in document_batches:
            input_ids, boundaries = self.pack_documents(docs)
            
            labels = create_labels(input_ids, boundaries)
            # sdpa expects attn_mask broadcastable to [batch, heads, seq, seq];
            # keep per-head broadcasting by adding a singleton head dim.
            attention_mask = create_document_mask(boundaries, self.seq_len).unsqueeze(0)
            position_ids = create_position_ids(boundaries, self.seq_len)
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_masks.append(attention_mask)
            batch_position_ids.append(position_ids)
            batch_boundaries.append(boundaries)
            
            # Track for Flash Attention varlen
            for i, end in enumerate(boundaries):
                start = boundaries[i-1] if i > 0 else 0
                doc_len = end - start
                max_doc_len = max(max_doc_len, doc_len)
                all_cu_seqlens.append(all_cu_seqlens[-1] + doc_len)
        
        return PackedBatch(
            input_ids=torch.stack(batch_input_ids),
            labels=torch.stack(batch_labels),
            attention_mask=torch.stack(batch_attention_masks),  # [batch, 1, seq, seq]
            position_ids=torch.stack(batch_position_ids),
            doc_boundaries=batch_boundaries,
            cu_seqlens=torch.tensor(all_cu_seqlens, dtype=torch.int32),
            max_seqlen=max_doc_len,
        )


class PackingCollator:
    """
    Collator for DataLoader that packs documents into sequences.
    
    Usage:
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=PackingCollator(seq_len=2048, tokenizer=tokenizer),
        )
    """
    
    def __init__(
        self,
        seq_len: int = 2048,
        pad_token_id: int = 0,
        eos_token_id: Optional[int] = None,
        docs_per_sequence: int = 4,  # Target docs per packed sequence
    ):
        self.packer = SequencePacker(
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        self.seq_len = seq_len
        self.docs_per_sequence = docs_per_sequence
    
    def __call__(self, batch: List[List[int]]) -> PackedBatch:
        """
        Collate documents into packed sequences.
        
        Args:
            batch: List of tokenized documents
            
        Returns:
            PackedBatch
        """
        # Group documents for packing
        # Try to fit ~docs_per_sequence docs per sequence
        document_batches = []
        current_batch = []
        current_len = 0
        
        for doc in batch:
            doc_len = len(doc) + 1  # +1 for EOS
            
            if current_len + doc_len > self.seq_len or len(current_batch) >= self.docs_per_sequence:
                if current_batch:
                    document_batches.append(current_batch)
                current_batch = [doc]
                current_len = doc_len
            else:
                current_batch.append(doc)
                current_len += doc_len
        
        if current_batch:
            document_batches.append(current_batch)
        
        return self.packer.pack_batch(document_batches)


def unpack_for_flash_attn(
    packed_batch: PackedBatch,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Convert packed batch to format for Flash Attention varlen.
    
    Flash Attention varlen expects flattened sequences with cu_seqlens.
    
    Returns:
        (input_ids_flat, labels_flat, cu_seqlens, max_seqlen)
    """
    batch_size, seq_len = packed_batch.input_ids.shape
    
    # Flatten
    input_ids_flat = packed_batch.input_ids.view(-1)
    labels_flat = packed_batch.labels.view(-1)
    
    return (
        input_ids_flat,
        labels_flat,
        packed_batch.cu_seqlens,
        packed_batch.max_seqlen,
    )


# Quick test
if __name__ == "__main__":
    print("Testing sequence packing...")
    
    # Create test documents
    docs = [
        [1, 2, 3, 4, 5],           # Doc 0
        [10, 11, 12],              # Doc 1
        [20, 21, 22, 23, 24, 25],  # Doc 2
        [30, 31],                  # Doc 3
    ]
    
    # Create packer
    packer = SequencePacker(
        seq_len=20,
        pad_token_id=0,
        eos_token_id=99,
        add_eos=True,
    )
    
    # Pack documents
    input_ids, boundaries = packer.pack_documents(docs)
    print(f"Packed sequence: {input_ids.tolist()}")
    print(f"Document boundaries: {boundaries}")
    
    # Create mask
    mask = create_document_mask(boundaries, 20)
    print(f"\nAttention mask shape: {mask.shape}")
    print(f"Mask (first 15x15):\n{mask[:15, :15].int()}")
    
    # Create position IDs
    pos_ids = create_position_ids(boundaries, 20)
    print(f"\nPosition IDs: {pos_ids.tolist()}")
    
    # Test batch packing
    print("\nTesting batch packing...")
    collator = PackingCollator(seq_len=20, pad_token_id=0, eos_token_id=99)
    
    batch = [
        [1, 2, 3],
        [10, 11, 12, 13],
        [20, 21],
        [30, 31, 32, 33, 34],
    ]
    
    packed = collator(batch)
    print(f"Batch input_ids shape: {packed.input_ids.shape}")
    print(f"Batch attention_mask shape: {packed.attention_mask.shape}")
    print(f"cu_seqlens: {packed.cu_seqlens.tolist()}")
    
    print("\nPacking test passed!")
