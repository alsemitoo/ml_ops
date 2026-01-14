"""Preprocessing transforms for LaTeX OCR images."""
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image


class FormulaResizePad:
    """Resize and pad formula images to fixed dimensions."""

    def __init__(self, target_height=128, max_width=640):
        """Initialize resize and pad transform.

        Args:
            target_height: Target height for images
            max_width: Maximum width for images (will be padded to this)
        """
        self.h = target_height
        self.w = max_width

    def __call__(self, img):
        """Apply resize and padding to image.

        Args:
            img: PIL Image to transform

        Returns:
            Padded PIL Image
        """
        original_w, original_h = img.size

        # 1. SCALING
        # Calculate new width to maintain aspect ratio
        scale_factor = self.h / original_h
        new_w = int(original_w * scale_factor)

        # 2. LIMITING
        # If the new width exceeds max_width (e.g., your 800px outliers),
        # we force it down to max_width. The height will remain 128.
        # This slightly distorts aspect ratio for outliers, but ensures no crashing.
        if new_w > self.w:
            new_w = self.w

        # Resize
        img = F.resize(img, (self.h, new_w))

        # 3. PADDING
        # Create white canvas
        padded_img = Image.new("RGB", (self.w, self.h), (255, 255, 255))
        padded_img.paste(img, (0, 0))

        return padded_img


def get_train_transform(target_height=128, max_width=640):
    """Get preprocessing transform pipeline for training (with augmentation).

    Args:
        target_height: Target height for images
        max_width: Maximum width for images

    Returns:
        Compose transform with augmentation
    """
    return transforms.Compose([
        FormulaResizePad(target_height=target_height, max_width=max_width),
        # Augmentation transforms
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_val_test_transform(target_height=128, max_width=640):
    """Get preprocessing transform pipeline for validation/test (no augmentation).

    Args:
        target_height: Target height for images
        max_width: Maximum width for images

    Returns:
        Compose transform without augmentation
    """
    return transforms.Compose([
        FormulaResizePad(target_height=target_height, max_width=max_width),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])