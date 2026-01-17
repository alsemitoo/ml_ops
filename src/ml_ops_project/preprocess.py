"""Preprocessing transforms for LaTeX OCR images."""

from loguru import logger
from PIL import Image
from torchvision import transforms


class FormulaResizePad:
    """Resize and pad formula images to fixed dimensions.

    This transform resizes images to a target height while maintaining aspect ratio,
    then pads them to a fixed width with white space.
    """

    def __init__(self, target_height: int = 128, max_width: int = 640) -> None:
        """Initialize resize and pad transform.

        Args:
            target_height: Target height for images (default: 128)
            max_width: Maximum width for images, pads to this (default: 640)
        """
        self.h = target_height
        self.w = max_width
        logger.debug(f"FormulaResizePad initialized: target_height={target_height}, max_width={max_width}")

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply resize and padding to image.

        Args:
            img: PIL Image to transform

        Returns:
            Padded PIL Image with shape (max_width, target_height)
        """
        original_w, original_h = img.size

        # 1. SCALING - Calculate new width to maintain aspect ratio
        scale_factor = self.h / original_h
        new_w = int(original_w * scale_factor)

        # 2. LIMITING - Clamp width if it exceeds max_width
        if new_w > self.w:
            logger.debug(f"Image width {new_w} exceeds max_width {self.w}, clamping...")
            new_w = self.w

        # Resize
        img = img.resize((new_w, self.h), resample=Image.Resampling.BILINEAR)

        # 3. PADDING - Create white canvas and paste resized image
        padded_img = Image.new("RGB", (self.w, self.h), (255, 255, 255))
        padded_img.paste(img, (0, 0))

        return padded_img


def get_train_transform(
    target_height: int = 128,
    max_width: int = 640,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> transforms.Compose:
    """Get preprocessing transform pipeline for training (with augmentation).

    Args:
        target_height: Target height for images (default: 128)
        max_width: Maximum width for images (default: 640)
        brightness: Brightness augmentation magnitude (default: 0.2)
        contrast: Contrast augmentation magnitude (default: 0.2)

    Returns:
        Compose transform with augmentation and normalization
    """
    logger.info(f"Creating train transform: h={target_height}, w={max_width}")
    return transforms.Compose(
        [
            FormulaResizePad(target_height=target_height, max_width=max_width),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_val_test_transform(
    target_height: int = 128,
    max_width: int = 640,
) -> transforms.Compose:
    """Get preprocessing transform pipeline for validation/test (no augmentation).

    Args:
        target_height: Target height for images (default: 128)
        max_width: Maximum width for images (default: 640)

    Returns:
        Compose transform without augmentation, only normalization
    """
    logger.info(f"Creating val/test transform: h={target_height}, w={max_width}")
    return transforms.Compose(
        [
            FormulaResizePad(target_height=target_height, max_width=max_width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


if __name__ == "__main__":
    logger.info("Testing preprocessing transforms...")
    train_tf = get_train_transform(target_height=128, max_width=640)
    val_tf = get_val_test_transform(target_height=128, max_width=640)
    logger.success("Preprocessing transforms initialized successfully")
