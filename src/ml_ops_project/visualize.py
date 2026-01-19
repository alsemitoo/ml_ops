"""Visualization utilities for training and evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger


def plot_training_statistics(statistics: dict[str, list[float]], output_path: Path) -> None:
    """Create and save training statistics plots.

    Args:
        statistics: Dictionary containing train/val loss and accuracy lists
        output_path: Path to save the plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(statistics["train_loss"])
    axs[0, 0].set_title("Train Loss")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(statistics["train_accuracy"])
    axs[0, 1].set_title("Train Accuracy")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Accuracy")

    axs[1, 0].plot(statistics["val_loss"])
    axs[1, 0].set_title("Validation Loss")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Loss")

    axs[1, 1].plot(statistics["val_accuracy"])
    axs[1, 1].set_title("Validation Accuracy")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    fig.savefig(output_path)
    logger.success(f"Training statistics saved to {output_path}")
