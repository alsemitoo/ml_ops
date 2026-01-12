from pathlib import Path
import json

from hydra import main as hydra_main
from loguru import logger
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from omegaconf import DictConfig


class MyDataset(Dataset):
    """LaTeX OCR Dataset for image-to-text tasks."""

    def __init__(self, data_path: Path = Path("data/raw/small_train")) -> None:
        """Initialize the dataset by loading from local files.

        Args:
            data_path: Path to the dataset folder containing images/ and labels.json
        """
        self.data_path = data_path
        self.images_path = data_path / "images"
        self.labels_file = data_path / "labels.json"

        # Load labels
        if self.labels_file.exists():
            with open(self.labels_file, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            logger.info(f"Loaded {len(self.labels)} samples from {data_path}")
        else:
            self.labels = []
            logger.info(f"No data found at {data_path}. Run download_data first.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Dictionary containing image and text
        """
        label_data = self.labels[index]
        image_path = self.images_path / label_data["image_file"]
        image = Image.open(image_path)

        return {"image": image, "text": label_data["text"]}


def download_data(cfg: DictConfig) -> None:
    """Download the LaTeX OCR dataset from HuggingFace and save locally.

    Args:
        cfg: Configuration object containing name, split, and output_path
    """
    name = cfg.name
    split = cfg.split
    output_path = Path(cfg.output_path)
    logger.info(f"Downloading LaTeX_OCR dataset (name={name}, split={split})...")
    dataset = load_dataset("linxy/LaTeX_OCR", name=name, split=split)

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
    logger.info(f"Dataset ready with {len(dataset)} samples!")
    logger.info(f"Sample: {dataset[0]['text'][:50]}...")


@hydra_main(config_path="../../configs", config_name="data", version_base=None)
def main(cfg: DictConfig) -> None:
    download_data(cfg)


if __name__ == "__main__":
    main()
