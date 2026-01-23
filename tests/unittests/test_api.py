import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from fastapi import HTTPException, UploadFile
from PIL import Image

from ml_ops_project.api import _init_model_artifacts_if_needed, beam_search_prediction, model_artifacts, predict
from ml_ops_project.tokenizer import LaTeXTokenizer


@pytest.fixture
def mock_tokenizer() -> LaTeXTokenizer:
    """Create a mock tokenizer for testing."""
    tokenizer = LaTeXTokenizer()
    tokenizer.vocab = {
        "<START>": 1,
        "<END>": 2,
        "x": 3,
        "+": 4,
        "y": 5,
        "<UNK>": 0,
    }
    tokenizer.idx_to_token = {v: k for k, v in tokenizer.vocab.items()}
    return tokenizer


class MockModel(torch.nn.Module):
    """Mock model that predicts END token on second position."""

    def __init__(self, vocab_size: int = 6) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, image: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Generate logits with high confidence for END token."""
        batch_size = image.size(0)
        seq_len = seq.size(1)
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[:, -1, 2] = 100
        return logits


class MockModelNoParams(torch.nn.Module):
    """Mock model with no parameters (tests StopIteration handling)."""

    def __init__(self, vocab_size: int = 6) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, image: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Generate logits with high confidence for END token."""
        batch_size = image.size(0)
        seq_len = seq.size(1)
        logits = torch.zeros(batch_size, seq_len, self.vocab_size)
        logits[:, -1, 2] = 100
        return logits


def test_beam_search_removes_start_token(mock_tokenizer: LaTeXTokenizer) -> None:
    """Test that START token is removed from output."""
    mock_model = MockModel(vocab_size=len(mock_tokenizer.vocab))
    mock_image = torch.randn(1, 3, 128, 640)

    result = beam_search_prediction(mock_model, mock_image, mock_tokenizer, beam_width=1, max_len=5)

    assert "<START>" not in result


def test_beam_search_removes_end_token(mock_tokenizer: LaTeXTokenizer) -> None:
    """Test that END token is removed from output."""
    mock_model = MockModel(vocab_size=len(mock_tokenizer.vocab))
    mock_image = torch.randn(1, 3, 128, 640)

    result = beam_search_prediction(mock_model, mock_image, mock_tokenizer, beam_width=1, max_len=5)

    assert "<END>" not in result


def test_beam_search_with_no_model_parameters(mock_tokenizer: LaTeXTokenizer) -> None:
    """Test beam search with model that has no parameters (StopIteration handling)."""
    mock_model = MockModelNoParams(vocab_size=len(mock_tokenizer.vocab))
    mock_image = torch.randn(1, 3, 128, 640)

    result = beam_search_prediction(mock_model, mock_image, mock_tokenizer, beam_width=1, max_len=5)

    assert isinstance(result, str)


def test_beam_search_returns_string(mock_tokenizer: LaTeXTokenizer) -> None:
    """Test that beam search returns a string."""
    mock_model = MockModel(vocab_size=len(mock_tokenizer.vocab))
    mock_image = torch.randn(1, 3, 128, 640)

    result = beam_search_prediction(mock_model, mock_image, mock_tokenizer, beam_width=1, max_len=5)

    assert isinstance(result, str)


def test_predict_with_missing_model_artifacts() -> None:
    """Test predict function when model artifacts are not loaded."""
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    mock_file = AsyncMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=img_byte_arr.getvalue())
    mock_file.filename = "test.png"

    with patch("ml_ops_project.api.model_artifacts", {"transform": None, "model": None, "tokenizer": None}):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(predict(mock_file))
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Model artifacts not loaded"


def test_predict_with_inference_failure() -> None:
    """Test predict function when beam_search_prediction raises an exception."""
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    mock_file = AsyncMock(spec=UploadFile)
    mock_file.read = AsyncMock(return_value=img_byte_arr.getvalue())
    mock_file.filename = "test.png"

    with patch("ml_ops_project.api.beam_search_prediction", side_effect=Exception("Inference failed")):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(predict(mock_file))
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to process image"


def test_init_model_artifacts_with_missing_vocab_file() -> None:
    """Test _init_model_artifacts_if_needed when vocab file doesn't exist."""
    model_artifacts.clear()

    with (
        patch("ml_ops_project.api.VOCAB_PATH") as mock_vocab_path,
        patch("ml_ops_project.api.MODEL_PATH") as mock_model_path,
    ):
        mock_vocab_path.exists.return_value = False
        mock_model_path.exists.return_value = False

        _init_model_artifacts_if_needed()

        assert "tokenizer" in model_artifacts
        assert model_artifacts["tokenizer"].vocab == {"<PAD>": 0, "<START>": 1, "<END>": 2}
        assert "model" in model_artifacts
        assert "transform" in model_artifacts
