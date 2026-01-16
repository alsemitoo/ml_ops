import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
from src.ml_ops_project.model import Im2LatexModel


# --- MOCKING HELPERS ---
def get_mock_resnet():
    """Creates a mock ResNet that returns a list of dummy layers when children() is called."""
    mock_resnet = MagicMock()
    # Create dummy layers to mimic list(resnet.children())[:-3]
    # We need enough layers so that [:-3] doesn't slice to empty
    mock_layers = [nn.Identity() for _ in range(10)]
    mock_resnet.children.return_value = mock_layers
    return mock_resnet


# --- FIXTURES ---
@pytest.fixture
def model_params():
    return {
        "vocab_size": 100,
        "d_model": 32,  # Small dimension for speed
        "nhead": 2,
        "num_decoder_layers": 1
    }


@pytest.fixture
def mock_backbone_model(model_params):
    """Returns model with mocked ResNet to avoid downloading weights."""
    with patch("torchvision.models.resnet18") as mock_resnet_fn:
        # Setup the mock to return a dummy backbone structure
        mock_resnet_fn.return_value = get_mock_resnet()
        
        model = Im2LatexModel(**model_params)
        
        # FIX: Use stride=16 to simulate ResNet downsampling
        # Input: (Batch, 3, 224, 224) -> Output: (Batch, 256, 14, 14)
        # 14 * 14 = 196 patches (which fits within the 320 limit)
        model.backbone = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=16, padding=1)
        )
        return model


# --- TESTS ---

def test_model_initialization(mock_backbone_model, model_params):
    """Test if model initializes with correct parameters."""
    assert mock_backbone_model.embedding.num_embeddings == model_params["vocab_size"]
    assert mock_backbone_model.fc_out.out_features == model_params["vocab_size"]
    # Check if parameters are registered
    assert sum(p.numel() for p in mock_backbone_model.parameters()) > 0


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [10, 20])
def test_model_forward_shape(mock_backbone_model, batch_size, seq_len, model_params):
    """Test that output shape matches (Batch, Seq_Len, Vocab_Size)."""
    # Create dummy inputs
    # Image: (Batch, 3, H, W)
    images = torch.randn(batch_size, 3, 224, 224)
    # Target: (Batch, Seq_Len)
    tgt_text = torch.randint(0, model_params["vocab_size"], (batch_size, seq_len))

    output = mock_backbone_model(images, tgt_text)

    # Assertions
    expected_shape = (batch_size, seq_len, model_params["vocab_size"])
    assert output.shape == expected_shape
    assert output.dtype == torch.float32


def test_causal_mask_structure():
    """Verify the causal mask is strictly upper triangular with -inf."""
    mask = Im2LatexModel._generate_square_subsequent_mask(4)
    
    # Shape check
    assert mask.shape == (4, 4)
    
    # Check diagonal and below is 0.0 (visible)
    # Note: The implementation might use -inf for masked and 0 for visible
    assert mask[0, 0] == 0.0
    assert mask[1, 0] == 0.0
    assert mask[1, 1] == 0.0
    
    # Check above diagonal is -inf (masked)
    assert mask[0, 1] == float("-inf")
    assert mask[0, 2] == float("-inf")
    assert mask[0, 3] == float("-inf")


def test_forward_pass_flow(mock_backbone_model, model_params):
    """Test that gradients can flow (smoke test for connectivity)."""
    images = torch.randn(2, 3, 64, 64)
    tgt_text = torch.randint(0, model_params["vocab_size"], (2, 10))
    
    output = mock_backbone_model(images, tgt_text)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are populated for a key layer
    assert mock_backbone_model.fc_out.weight.grad is not None


def test_error_on_wrong_image_channels(mock_backbone_model):
    """Test if model fails gracefully/predictably with wrong image channels."""
    # Input has 1 channel (grayscale) instead of 3
    images = torch.randn(1, 1, 64, 64)
    tgt_text = torch.randint(0, 10, (1, 5))
    
    # Expect a RuntimeError due to Conv2d mismatch
    with pytest.raises(RuntimeError) as excinfo:
        mock_backbone_model(images, tgt_text)
    
    # The error usually mentions channel mismatch
    assert "Expected" in str(excinfo.value) or "channels" in str(excinfo.value)