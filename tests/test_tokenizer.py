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
    """Tokenizer initializes with special token strings even when vocab is empty.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
    """
    # vocab is empty initially (by design)
    assert tokenizer.vocab_size == 0

    # but the special token strings should exist as attributes
    assert tokenizer.pad_token == "<PAD>"
    assert tokenizer.unk_token == "<UNK>"
    assert tokenizer.start_token == "<START>"
    assert tokenizer.end_token == "<END>"


def test_build_vocab_success(tokenizer: LaTeXTokenizer, make_labels_file):
    """`build_vocab` creates special tokens and includes tokens from labels.json.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` file for testing.
    """
    # Create a labels.json with some sample data
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": r"\frac{1}{2}"},
        ]
    )

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
    """Missing labels.json path raises `FileNotFoundError` during `build_vocab`.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        tmp_path: Temporary directory to construct a missing file path.
    """
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        tokenizer.build_vocab(missing)


def test_build_vocab_malformed_json_raises(tokenizer: LaTeXTokenizer, tmp_path: Path):
    """Malformed JSON content causes `json.JSONDecodeError` in `build_vocab`.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        tmp_path: Temporary directory used to write malformed `labels.json`.
    """
    labels_path = tmp_path / "labels.json"
    labels_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        tokenizer.build_vocab(labels_path)


# --------- Encoder Tests ---------
def test_encode_adds_special_tokens_by_default(tokenizer: LaTeXTokenizer, make_labels_file):
    """By default, `encode` wraps output with START and END tokens.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a b"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    encoded = tokenizer.encode("a b")

    # Should be [START, a, b, END]
    assert len(encoded) == 4
    assert encoded[0] == tokenizer.vocab[tokenizer.start_token]
    assert encoded[-1] == tokenizer.vocab[tokenizer.end_token]


def test_encode_without_special_tokens(tokenizer: LaTeXTokenizer, make_labels_file):
    """`encode` omits START and END when `add_special_tokens=False`.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a b"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    encoded = tokenizer.encode("a b", add_special_tokens=False)

    # Should be [a, b] without START/END
    assert len(encoded) == 2
    assert encoded[0] == tokenizer.vocab["a"]
    assert encoded[1] == tokenizer.vocab["b"]


def test_encode_unknown_tokens_map_to_unk(tokenizer: LaTeXTokenizer, make_labels_file):
    """Unknown tokens in input map to UNK index.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    encoded = tokenizer.encode("a unknown_token", add_special_tokens=False)

    # Should be [a, UNK]
    assert len(encoded) == 2
    assert encoded[0] == tokenizer.vocab["a"]
    assert encoded[1] == tokenizer.vocab[tokenizer.unk_token]


def test_encode_with_empty_vocab_raises_keyerror(tokenizer: LaTeXTokenizer):
    """When vocab is empty, `encode` raises `KeyError` due to missing UNK fallback.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture with no vocab built.
    """
    assert tokenizer.vocab_size == 0

    # Should raise KeyError when trying to access self.vocab[self.unk_token]
    with pytest.raises(KeyError):
        tokenizer.encode("a b", add_special_tokens=False)


# --------- Decoder Tests ---------
def test_decode_skips_special_tokens(tokenizer: LaTeXTokenizer, make_labels_file):
    """`decode` drops special tokens by default when `skip_special_tokens=True`.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a b"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    # Build a sequence: <START> a b <END>
    indices = [
        tokenizer.vocab[tokenizer.start_token],
        tokenizer.vocab["a"],
        tokenizer.vocab["b"],
        tokenizer.vocab[tokenizer.end_token],
    ]

    decoded = tokenizer.decode(indices)

    assert decoded == "a b"


def test_decode_keeps_special_tokens(tokenizer: LaTeXTokenizer, make_labels_file):
    """`decode` retains special tokens when `skip_special_tokens=False`.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    indices = [
        tokenizer.vocab[tokenizer.start_token],
        tokenizer.vocab["a"],
        tokenizer.vocab[tokenizer.end_token],
    ]

    decoded = tokenizer.decode(indices, skip_special_tokens=False)

    assert decoded == "<START> a <END>"


def test_decode_unknown_index(tokenizer: LaTeXTokenizer, make_labels_file):
    """Unknown indices map to `<UNK>`; default skip removes it from output.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    unknown_idx = 9999  # not in vocab

    decoded = tokenizer.decode([unknown_idx])

    # default skips special tokens, and <UNK> is considered special → removed
    assert decoded == ""

    decoded2 = tokenizer.decode([unknown_idx], skip_special_tokens=False)
    assert decoded2 == "<UNK>"


# --------- Edge Case Tests ---------
def test_encode_empty_text(tokenizer: LaTeXTokenizer, make_labels_file):
    """Encoding empty string returns only special tokens when enabled.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    encoded = tokenizer.encode("")

    # Should be [START, END]
    assert len(encoded) == 2
    assert encoded[0] == tokenizer.vocab[tokenizer.start_token]
    assert encoded[1] == tokenizer.vocab[tokenizer.end_token]


def test_encode_single_token(tokenizer: LaTeXTokenizer, make_labels_file):
    """Encoding single token wraps it with START and END tokens by default.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    encoded = tokenizer.encode("a")

    # Should be [START, a, END]
    assert len(encoded) == 3
    assert encoded[0] == tokenizer.vocab[tokenizer.start_token]
    assert encoded[1] == tokenizer.vocab["a"]
    assert encoded[2] == tokenizer.vocab[tokenizer.end_token]


def test_decode_empty_indices(tokenizer: LaTeXTokenizer):
    """Decoding empty index list returns empty string.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
    """
    decoded = tokenizer.decode([])

    assert decoded == ""


def test_decode_single_index(tokenizer: LaTeXTokenizer, make_labels_file):
    """Decoding single regular token returns that token.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    indices = [tokenizer.vocab["a"]]
    decoded = tokenizer.decode(indices)

    assert decoded == "a"


def test_encode_decode_round_trip(tokenizer: LaTeXTokenizer, make_labels_file):
    """Encoding then decoding recovers the original text.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a b c"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    text = "a b c"
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)

    assert decoded == text


# --------- Pad/Unk Index Tests ---------
def test_get_pad_idx_without_vocab_returns_zero(tokenizer: LaTeXTokenizer):
    """When vocab is empty, `get_pad_idx` should fall back to 0.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
    """
    assert tokenizer.vocab_size == 0
    assert tokenizer.get_pad_idx() == 0


def test_get_pad_idx_with_vocab(tokenizer: LaTeXTokenizer, make_labels_file):
    """After building vocab, `get_pad_idx` returns the actual PAD index.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "a b"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    pad_idx = tokenizer.get_pad_idx()
    assert tokenizer.pad_token in tokenizer.vocab
    assert pad_idx == tokenizer.vocab[tokenizer.pad_token]


def test_get_unk_idx_without_vocab_returns_one(tokenizer: LaTeXTokenizer):
    """When vocab is empty, `get_unk_idx` should fall back to 1.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
    """
    assert tokenizer.vocab_size == 0
    assert tokenizer.get_unk_idx() == 1


def test_get_unk_idx_with_vocab(tokenizer: LaTeXTokenizer, make_labels_file):
    """After building vocab, `get_unk_idx` returns the actual UNK index.

    Args:
        tokenizer: Fresh `LaTeXTokenizer` fixture.
        make_labels_file: Helper that writes a `labels.json` with simple tokens.
    """
    labels_path = make_labels_file(
        [
            {"image_file": "img.png", "text": "x y"},
        ]
    )

    tokenizer.build_vocab(labels_path)

    unk_idx = tokenizer.get_unk_idx()
    assert tokenizer.unk_token in tokenizer.vocab
    assert unk_idx == tokenizer.vocab[tokenizer.unk_token]
