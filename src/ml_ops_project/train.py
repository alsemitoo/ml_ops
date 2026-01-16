"""Training script for Image-to-LaTeX model."""
import cProfile
import pstats
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from ml_ops_project.data import MyDataset
from ml_ops_project.model import Im2LatexModel
from ml_ops_project.preprocess import get_train_transform, get_val_test_transform
from ml_ops_project.tokenizer import LaTeXTokenizer
from ml_ops_project.visualize import plot_training_statistics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function to handle variable-length sequences with padding.

    Args:
        batch: List of (image, label_tensor) tuples

    Returns:
        Tuple of (images, labels) where labels are padded to same length
    """
    images_list, labels_list = zip(*batch)
    images = torch.stack(list(images_list))

    max_len = max(len(label) for label in labels_list)
    padded_labels = []
    for label in labels_list:
        padding = torch.zeros(max_len - len(label), dtype=torch.long)
        padded_labels.append(torch.cat([label, padding]))

    labels = torch.stack(padded_labels)
    return images, labels


def prepare_datasets(
    data_path: Path, tokenizer: LaTeXTokenizer
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Prepare train, validation, and test datasets.

    Args:
        data_path: Path to dataset directory
        tokenizer: Tokenizer instance

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create datasets with appropriate transforms for each split
    logger.info("Loading datasets...")
    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()

    # Create base dataset to get indices
    base_dataset = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=None,
    )

    # Split dataset: 70% train, 15% val, 15% test
    total_size = len(base_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Get indices from subsets
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    # Create separate datasets with appropriate transforms
    train_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=train_transform,
    )
    train_dataset = torch.utils.data.Subset(train_dataset_base, train_indices)

    val_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=val_test_transform,
    )
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)

    test_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=val_test_transform,
    )
    test_dataset = torch.utils.data.Subset(test_dataset_base, test_indices)

    logger.info(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def train_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    pad_idx: int,
    epoch: int,
) -> tuple[list[float], list[float]]:
    """Train the model for one epoch.

    Args:
        model: The Image-to-LaTeX model
        dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        vocab_size: Size of the vocabulary
        pad_idx: Padding token index
        epoch: Current epoch number

    Returns:
        Tuple of (epoch_train_loss, epoch_train_acc)
    """
    # Training phase
    model.train()
    epoch_train_loss = []
    epoch_train_acc = []

    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Teacher forcing: input is labels[:-1], target is labels[1:]
        # For each sequence, we predict the next token given previous tokens
        tgt_input = labels[:, :-1]  # Remove last token for input
        tgt_output = labels[:, 1:]  # Remove first token for target
        optimizer.zero_grad()
        y_pred = model(images, tgt_input)  # (Batch, Seq_Len-1, Vocab_Size)

        # Reshape for loss: (Batch * Seq_Len-1, Vocab_Size) and (Batch * Seq_Len-1,)
        loss = loss_fn(y_pred.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())

        # Calculate accuracy (excluding padding)
        pred_tokens = y_pred.argmax(dim=2)
        mask = tgt_output != pad_idx
        correct = (pred_tokens == tgt_output) * mask
        accuracy = correct.sum().float() / mask.sum().float()
        epoch_train_acc.append(accuracy.item())

        if i % 100 == 0:
            logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, acc: {accuracy.item():.4f}")

    return epoch_train_loss, epoch_train_acc


def validate_epoch(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    vocab_size: int,
    pad_idx: int,
) -> tuple[list[float], list[float]]:
    """Validate the model for one epoch.

    Args:
        model: The Image-to-LaTeX model
        dataloader: DataLoader for validation data
        loss_fn: Loss function
        vocab_size: Size of the vocabulary
        pad_idx: Padding token index
        epoch: Current epoch number

    Returns:
        Tuple of (epoch_val_loss, epoch_val_acc)
    """
    # Validation phase
    model.eval()
    epoch_val_loss = []
    epoch_val_acc = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            y_pred = model(images, tgt_input)

            loss = loss_fn(y_pred.reshape(-1, vocab_size), tgt_output.reshape(-1))

            epoch_val_loss.append(loss.item())

            pred_tokens = y_pred.argmax(dim=2)
            mask = tgt_output != pad_idx
            correct = (pred_tokens == tgt_output) * mask
            accuracy = correct.sum().float() / mask.sum().float()
            epoch_val_acc.append(accuracy.item())

    return epoch_val_loss, epoch_val_acc


@hydra_main(config_path="../../configs", config_name="train", version_base=None)
def train(cfg: DictConfig):
    """Train the Image-to-LaTeX model.

    Args:
        cfg: Hydra configuration object containing training, model, and data parameters
    """
    # Configure logger
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/train.log", rotation="10 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    train_cfg = cfg.training
    model_cfg = cfg.model

    logger.info(
        f"Starting training with epochs={train_cfg.epochs}, batch_size={train_cfg.batch_size}, data_path={train_cfg.data_path}"
    )

    profiler = cProfile.Profile()
    profiler.enable()

    data_path = Path(train_cfg.data_path)

    # Build tokenizer from labels
    logger.info("Building tokenizer from labels...")
    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(data_path / "labels.json")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.get_pad_idx()

    logger.info(f"Vocabulary size: {vocab_size}")

    train_dataset, val_dataset, test_dataset = prepare_datasets(data_path, tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Initialize model
    logger.info(f"Initializing model on device: {DEVICE}")
    model = Im2LatexModel(
        vocab_size=vocab_size,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_decoder_layers=model_cfg.num_decoder_layers,
    )
    model.to(DEVICE)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss function (ignore padding tokens)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    statistics: dict[str, list[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(train_cfg.epochs):
        # Training phase
        epoch_train_loss, epoch_train_acc = train_epoch(
            model, train_dataloader, loss_fn, optimizer, vocab_size, pad_idx, epoch
        )

        epoch_val_loss, epoch_val_acc = validate_epoch(model, val_dataloader, loss_fn, vocab_size, pad_idx)

        # Record statistics
        statistics["train_loss"].extend(epoch_train_loss)
        statistics["train_accuracy"].extend(epoch_train_acc)
        statistics["val_loss"].extend(epoch_val_loss)
        statistics["val_accuracy"].extend(epoch_val_acc)

        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_acc = np.mean(epoch_train_acc)
        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_acc = np.mean(epoch_val_acc)

        logger.info(f"Epoch {epoch} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

    logger.success("Training complete")

    profiler.disable()
    # TODO: Revise this
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumulative")
    # Path("reports/profiling").mkdir(parents=True, exist_ok=True)
    # stats.dump_stats("reports/profiling/train_profile.prof")
    # with open("reports/profiling/train_profile.txt", "w") as f:
    #     stats.stream = f
    #     stats.print_stats(50)

    # Save model and tokenizer
    Path("models").mkdir(exist_ok=True)
    logger.info("Saving model and tokenizer...")
    torch.save(model.state_dict(), "models/model.pth")
    torch.save(tokenizer.vocab, "models/vocab.pt")
    logger.success("Model and vocabulary saved to models/")

    plot_training_statistics(statistics, Path("logs/training_statistics.png"))


if __name__ == "__main__":
    train()
