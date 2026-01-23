"""Generate artificially drifted images for testing data drift detection.

Usage:
    uv run python scripts/create_drifted_data.py
"""

import os
from pathlib import Path

from PIL import Image, ImageEnhance


def create_drifted_data(
    input_dir: str | Path, output_dir: str | Path, brightness_factor: float = 0.2, max_images: int | None = None
) -> None:
    """Create drifted images by reducing brightness.

    Args:
        input_dir: Source image directory.
        output_dir: Destination for drifted images.
        brightness_factor: Brightness multiplier (0.2 = 80% darker).
        max_images: Maximum number of images to process, None for all.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in input_dir.glob("*.png") if f.is_file()])

    if max_images:
        image_files = image_files[:max_images]

    if not image_files:
        print(f"No PNG images found in {input_dir}")
        return

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            enhancer = ImageEnhance.Brightness(img)
            drifted_img = enhancer.enhance(brightness_factor)
            drifted_img.save(output_dir / img_path.name)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images...")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print(f"Created {len(image_files)} drifted images in {output_dir}")


if __name__ == "__main__":
    create_drifted_data(
        input_dir="data/raw/default_train/images",
        output_dir="data/drifted_current/images",
        brightness_factor=0.2,
        max_images=6318,  # Match your dataset size
    )
