from pathlib import Path

import pytest
from PIL import Image

from ml_ops_project.visualize import plot_training_statistics


def test_plot_training_statistics_creates_file(tmp_path):
    """Test that plot_training_statistics creates a valid PNG file."""
    # Arrange: minimal statistics with 5 iterations each
    statistics = {
        "train_loss": [0.5, 0.4, 0.3, 0.25, 0.2],
        "train_accuracy": [0.6, 0.7, 0.75, 0.8, 0.85],
        "val_loss": [0.6, 0.5, 0.45, 0.4, 0.35],
        "val_accuracy": [0.55, 0.65, 0.7, 0.75, 0.8],
    }

    output_path = tmp_path / "plots" / "training_stats_test.png"

    plot_training_statistics(statistics, output_path)

    # Assert: file exists and is non-empty
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Assert: file is a valid image
    img = Image.open(output_path)
    assert img.format == "PNG"
    assert img.size[0] > 0 and img.size[1] > 0
