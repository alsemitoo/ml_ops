# tests/test_tokenizer.py
import json
from pathlib import Path

import pytest

from ml_ops_project.tokenizer import LaTeXTokenizer

# ---------- Fixtures ----------
@pytest.fixture
def tokenizer() -> LaTeXTokenizer:
    """Fresh tokenizer instance for each test."""
    return LaTeXTokenizer()


@pytest.fixture
def make_labels_file(tmp_path: Path):
    """
    Create a minimal labels.json file. Returns a function so tests can customize content.
    """
    def _make(labels) -> Path:
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        return labels_path

    return _make



# ---------- Tests ----------
def test_create_tokenizer_has_special_tokens(tokenizer: LaTeXTokenizer):
    # vocab is empty initially (by design)
    assert tokenizer.vocab_size == 0

    # but the special token strings should exist as attributes
    assert tokenizer.pad_token == "<PAD>"
    assert tokenizer.unk_token == "<UNK>"
    assert tokenizer.start_token == "<START>"
    assert tokenizer.end_token == "<END>"


def test_build_vocab_success(tokenizer: LaTeXTokenizer, make_labels_file):
    labels_path = make_labels_file([
        {"image_file": "img.png", "text": r"\frac{1}{2}"},
    ])

    tokenizer.build_vocab(labels_path)

    # after build_vocab, vocab should contain special tokens
    assert tokenizer.vocab_size >= 4
    assert tokenizer.pad_token in tokenizer.vocab
    assert tokenizer.unk_token in tokenizer.vocab
    assert tokenizer.start_token in tokenizer.vocab
    assert tokenizer.end_token in tokenizer.vocab

    # and token from labels should be in vocab
    assert r"\frac{1}{2}" in tokenizer.vocab


def test_build_vocab_missing_file_raises(tokenizer: LaTeXTokenizer, tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        tokenizer.build_vocab(missing)


def test_build_vocab_malformed_json_raises(tokenizer: LaTeXTokenizer, tmp_path: Path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        tokenizer.build_vocab(labels_path)
