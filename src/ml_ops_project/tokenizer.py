"""Tokenizer module for LaTeX OCR dataset.

This module provides functionality to build a vocabulary from LaTeX labels
by extracting all unique tokens separated by spaces.
"""
import json
from pathlib import Path

from loguru import logger


class LaTeXTokenizer:
    """Tokenizer for LaTeX formulas that extracts tokens from space-separated strings."""

    def __init__(self, vocab: dict[str, int] | None = None):
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
            logger.error(f"Labels file not found: {labels_file}")
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        logger.info(f"Building vocabulary from {labels_file}")
        with open(labels_file, "r", encoding="utf-8") as f:
            labels = json.load(f)

        # Extract all unique tokens
        all_tokens: set[str] = set()
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

        logger.success(f"Built vocabulary with {len(self.vocab)} tokens from {len(labels)} labels")

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
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

        logger.debug(f"Encoded text with {len(indices)} tokens (special_tokens={add_special_tokens})")
        return indices

    def decode(self, indices: list[int], skip_special_tokens: bool = True) -> str:
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
        
        logger.debug(f"Decoded {len(indices)} indices to {len(tokens)} tokens (skip_special={skip_special_tokens})")
        return " ".join(tokens)

    def get_pad_idx(self) -> int:
        """Get the padding token index.

        Returns:
            Index of the padding token.
        """
        return self.vocab.get(self.pad_token, 0)

    def get_unk_idx(self) -> int:
        """Get the unknown token index.

        Returns:
            Index of the unknown token.
        """
        return self.vocab.get(self.unk_token, 1)


if __name__ == "__main__":
    logger.info("Testing LaTeX tokenizer...")
    
    # Create a test tokenizer
    tokenizer = LaTeXTokenizer()
    logger.info(f"Created tokenizer with vocab size: {tokenizer.vocab_size}")
    
    # Test encode/decode
    test_text = "\\frac{x}{y} + \\sqrt{z}"
    encoded = tokenizer.encode(test_text)
    logger.info(f"Encoded '{test_text}' to {len(encoded)} tokens")
    
    logger.success("Tokenizer test completed successfully")
