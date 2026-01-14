"""Tokenizer module for LaTeX OCR dataset.

This module provides functionality to build a vocabulary from LaTeX labels
by extracting all unique tokens separated by spaces.
"""
import json
from pathlib import Path
from typing import Dict, List, Set


class LaTeXTokenizer:
    """Tokenizer for LaTeX formulas that extracts tokens from space-separated strings."""

    def __init__(self, vocab: Dict[str, int] = None):
        """Initialize the tokenizer.

        Args:
            vocab: Optional pre-built vocabulary dictionary mapping tokens to indices.
                  If None, vocabulary must be built using build_vocab.
        """
        self.vocab = vocab or {}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()} if self.vocab else {}
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.start_token = "<START>"
        self.end_token = "<END>"

    def build_vocab(self, labels_file: Path) -> None:
        """Build vocabulary from labels.json file.

        Extracts all unique tokens by splitting labels on whitespace.

        Args:
            labels_file: Path to labels.json file containing label data.
        """
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)

        # Extract all unique tokens
        all_tokens: Set[str] = set()
        for item in labels:
            text = item.get("text", "")
            # Split on whitespace to get tokens
            tokens = text.split()
            all_tokens.update(tokens)

        # Create vocabulary with special tokens first
        special_tokens = [self.pad_token, self.unk_token, self.start_token, self.end_token]
        vocab_list = special_tokens + sorted(all_tokens)

        # Build token to index mapping
        self.vocab = {token: idx for idx, token in enumerate(vocab_list)}
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

        print(f"Built vocabulary with {len(self.vocab)} tokens")

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a text string into token indices.

        Args:
            text: Input LaTeX text (space-separated tokens).
            add_special_tokens: Whether to add START and END tokens.

        Returns:
            List of token indices.
        """
        tokens = text.split()
        indices = []
        
        if add_special_tokens:
            indices.append(self.vocab.get(self.start_token, self.vocab[self.unk_token]))
        
        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab[self.unk_token]))
        
        if add_special_tokens:
            indices.append(self.vocab.get(self.end_token, self.vocab[self.unk_token]))
        
        return indices

    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token indices back to text string.

        Args:
            indices: List of token indices.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def get_pad_idx(self) -> int:
        """Get the padding token index."""
        return self.vocab.get(self.pad_token, 0)

    def get_unk_idx(self) -> int:
        """Get the unknown token index."""
        return self.vocab.get(self.unk_token, 1)
