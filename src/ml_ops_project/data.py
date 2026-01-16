"""Data loading and dataset classes for LaTeX OCR."""

import json
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset
from hydra import main as hydra_main
from loguru import logger
from omegaconf import DictConfig
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
            logger.info(f"Loaded {len(self.labels)} samples from {data_path}")
        else:
            self.labels = []
            logger.warning(f"No data found at {data_path}. Run download_data first.")

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

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            raise

        # Apply preprocessing transform
        if self.transform:
            image = self.transform(image)

        # Encode text to token indices
        token_indices = self.tokenizer.encode(latex_text, add_special_tokens=True)
        label_tensor = torch.tensor(token_indices, dtype=torch.long)

        return image, label_tensor


@hydra_main(config_path="../../configs", config_name="data", version_base=None)
def download_data(cfg: DictConfig) -> None:
    """Download the LaTeX OCR dataset from HuggingFace and save locally.

    Args:
        name: Dataset variant (small, full, synthetic_handwrite, human_handwrite, human_handwrite_print)
        split: Dataset split (train, validation, test)
        output_path: Base path to save the dataset
    """

    download_cfg = cfg.download
    name = download_cfg.name
    split = download_cfg.split
    output_path = Path(download_cfg.output_path)

    logger.info(f"Downloading LaTeX_OCR dataset (name={name}, split={split})...")
    dataset = load_dataset("linxy/LaTeX_OCR", name=name, split=split)
    logger.info(f"Successfully loaded dataset with {len(dataset)} samples")

    # Create output directories
    dataset_folder = output_path / f"{name}_{split}"
    images_folder = dataset_folder / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # Save images and collect labels
    labels = []
    logger.info(f"Saving {len(dataset)} samples to {dataset_folder}...")

    for idx, item in enumerate(dataset):
        # Save image
        image_filename = f"image_{idx:06d}.png"
        image_path = images_folder / image_filename
        item["image"].save(image_path)

        # Store label info
        labels.append({"image_file": image_filename, "text": item["text"]})

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(dataset)} samples...")

    # Save labels as JSON
    labels_file = dataset_folder / "labels.json"
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    logger.success(f"✓ Dataset saved to {dataset_folder}")
    logger.info(f"  - Images: {images_folder}")
    logger.info(f"  - Labels: {labels_file}")
    logger.info(f"  - Total samples: {len(labels)}")


if __name__ == "__main__":
    download_data()
