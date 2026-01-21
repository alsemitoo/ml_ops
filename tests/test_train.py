from unittest.mock import MagicMock, patch

import pytest
import torch

from ml_ops_project.train import collate_fn, validate_epoch


@pytest.mark.parametrize("batch_size", [2, 4])
def test_collate_fn(batch_size):
    # Mock data: (image, label) tuples
    # Images: (1, 28, 28), Labels: Variable length tensors
    batch = [
        (torch.randn(1, 28, 28), torch.randint(0, 10, (10,))),  # Length 10
        (torch.randn(1, 28, 28), torch.randint(0, 10, (15,))),  # Length 15 (Max)
    ]

    images, labels = collate_fn(batch)

    # Assert shapes
    assert images.shape == (2, 1, 28, 28)
    assert labels.shape == (2, 15)  # Should match max length

    # Assert padding (short sequence should end with 0s)
    assert (labels[0, 10:] == 0).all()


from unittest.mock import MagicMock, patch

from ml_ops_project.train import prepare_datasets


@patch("ml_ops_project.train.MyDataset")
def test_prepare_datasets_splits(mock_dataset):
    # Setup mock to look like a dataset of 100 items
    mock_instance = MagicMock()
    mock_instance.__len__.return_value = 100
    mock_dataset.return_value = mock_instance

    mock_tokenizer = MagicMock()

    train, val, test = prepare_datasets("dummy/path", mock_tokenizer)

    # Check split sizes (70%, 15%, 15% of 100)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


import torch
import torch.nn as nn

from ml_ops_project.train import train_epoch


# Define a lightweight dummy model for testing
class DummyModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Define a simple layer so the model has parameters to optimize
        self.linear = nn.Linear(10, vocab_size)

    def forward(self, images, tgt_input):
        # Return random output of shape (Batch, Seq_Len, Vocab_Size)
        # Matches the expected output of your real model
        batch_size = images.shape[0]
        seq_len = tgt_input.shape[1]
        vocab_size = self.linear.out_features

        # Return a tensor that requires_grad (implicitly done by nn.Linear)
        return torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)


@patch("ml_ops_project.train.DEVICE", "cpu")
def test_train_epoch_smoke():
    vocab_size = 10
    batch_size = 2
    seq_len = 5

    # Initialize real objects (but small/dummy ones)
    model = DummyModel(vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Create dummy data matching the collate_fn output
    dummy_images = torch.randn(batch_size, 1, 28, 28)
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len + 1))

    # Mock dataloader as a simple list
    dataloader = [(dummy_images, dummy_labels)]

    # Create a Mock Scaler
    mock_scaler = MagicMock()

    # Run the function
    loss, acc = train_epoch(
        model=model,
        train_dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        vocab_size=vocab_size,
        pad_idx=0,
        epoch=0,
        scaler=mock_scaler,
    )

    # Assertions
    assert len(loss) == 1
    assert isinstance(loss[0], float)
    assert len(acc) == 1
    assert 0.0 <= acc[0] <= 1.0

    # Verify scaler was called
    assert mock_scaler.scale.called
    assert mock_scaler.step.called
    assert mock_scaler.update.called


@patch("ml_ops_project.train.DEVICE", "cpu")
def test_validate_epoch_smoke():
    """Validate that validate_epoch computes loss and accuracy for one batch."""
    vocab_size = 10
    pad_idx = 0
    batch_size = 2
    seq_len = 5

    model = DummyModel(vocab_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # Single-batch dataloader matching expected shapes
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    dataloader = [(images, labels)]

    val_loss, val_acc = validate_epoch(
        model=model,
        val_dataloader=dataloader,
        loss_fn=loss_fn,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
    )

    assert len(val_loss) == 1
    assert isinstance(val_loss[0], float)
    assert len(val_acc) == 1
    assert 0.0 <= val_acc[0] <= 1.0


@patch("ml_ops_project.train.DEVICE", "cpu")
@pytest.mark.xfail(strict=True, reason="Demonstration: this assertion is intentionally incorrect")
def test_validate_epoch_expected_fail():
    """Intentionally failing test to demonstrate xfail behavior."""
    vocab_size = 10
    pad_idx = 0
    batch_size = 2
    seq_len = 5

    model = DummyModel(vocab_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    dataloader = [(images, labels)]

    val_loss, val_acc = validate_epoch(
        model=model,
        val_dataloader=dataloader,
        loss_fn=loss_fn,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
    )

    # Wrong expectation: we only provided one batch, but we assert two
    assert len(val_loss) == 2
