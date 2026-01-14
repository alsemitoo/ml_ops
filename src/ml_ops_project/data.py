"""Data loading and dataset classes for LaTeX OCR."""
import json
from pathlib import Path
from typing import Callable

import torch
import typer
from datasets import load_dataset  # type: ignore
from PIL import Image
from torch.utils.data import Dataset

from ml_ops_project.tokenizer import LaTeXTokenizer


class MyDataset(Dataset):
    """LaTeX OCR Dataset for image-to-text tasks."""

    def __init__(
        self,
        data_path: Path,
        tokenizer: LaTeXTokenizer,
        transform: Callable | None = None,
    ) -> None:
        """Initialize the dataset by loading from local files.

        Args:
            data_path: Path to the dataset folder containing images/ and labels.json
            tokenizer: LaTeXTokenizer instance for encoding text
            transform: Optional transform pipeline for image preprocessing
        """
        self.data_path = Path(data_path)
        self.images_path = self.data_path / "images"
        self.labels_file = self.data_path / "labels.json"
        self.tokenizer = tokenizer
        self.transform = transform

        # Load labels
        if self.labels_file.exists():
            with open(self.labels_file, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            print(f"Loaded {len(self.labels)} samples from {data_path}")
        else:
            self.labels = []
            print(f"No data found at {data_path}. Run download_data first.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Returns:
            Tuple of (image_tensor, label_tensor) where:
            - image_tensor: Preprocessed image tensor
            - label_tensor: Token indices as a tensor
        """
        # Load image
        item = self.labels[idx]
        img_name = item["image_file"]
        latex_text = item["text"]

        image_path = self.images_path / img_name
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing transform
        if self.transform:
            image = self.transform(image)

        # Encode text to token indices
        token_indices = self.tokenizer.encode(latex_text, add_special_tokens=True)
        label_tensor = torch.tensor(token_indices, dtype=torch.long)

        return image, label_tensor


def download_data(
    name: str = "default",
    split: str = "train",
    output_path: Path = Path("data/raw"),
) -> None:
    """Download the LaTeX OCR dataset from HuggingFace and save locally.

    Args:
        name: Dataset variant (small, full, synthetic_handwrite, human_handwrite, human_handwrite_print)
        split: Dataset split (train, validation, test)
        output_path: Base path to save the dataset
    """
    print(f"Downloading LaTeX_OCR dataset (name={name}, split={split})...")
    dataset = load_dataset("linxy/LaTeX_OCR", name=name, split=split)

    # Create output directories
    dataset_folder = output_path / f"{name}_{split}"
    images_folder = dataset_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # Save images and collect labels
    labels = []
    print(f"Saving {len(dataset)} samples to {dataset_folder}...")

    for idx, item in enumerate(dataset):
        # Save image
        image_filename = f"image_{idx:06d}.png"
        image_path = images_folder / image_filename
        item["image"].save(image_path)

        # Store label info
        labels.append({"image_file": image_filename, "text": item["text"]})

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} samples...")

    # Save labels as JSON
    labels_file = dataset_folder / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"✓ Dataset saved to {dataset_folder}")
    print(f"  - Images: {images_folder}")
    print(f"  - Labels: {labels_file}")
    print(f"  - Total samples: {len(labels)}")
    print(f"Dataset ready with {len(dataset)} samples!")
    print(f"Sample: {dataset[0]['text'][:50]}...")


if __name__ == "__main__":
    typer.run(download_data)
