"""Simple data drift robustness testing for the LaTeX OCR model.

This module tests how robust the model is to image quality degradations
like blur, noise, and brightness changes.
"""

import torch
from loguru import logger
from torchvision import transforms


def blur_image(image: torch.Tensor, severity: int = 1) -> torch.Tensor:
    """Make image blurry (simulates out-of-focus camera).

    Args:
        image: Image tensor (C, H, W).
        severity: 1=mild blur, 2=moderate, 3=severe.

    Returns:
        Blurred image.
    """
    kernel_sizes = {1: 3, 2: 5, 3: 9}
    kernel_size = kernel_sizes.get(severity, 5)
    blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=2.0)
    return blur(image)


def add_noise(image: torch.Tensor, severity: int = 1) -> torch.Tensor:
    """Add random noise to image (simulates low-quality camera).

    Args:
        image: Image tensor (C, H, W).
        severity: 1=mild noise, 2=moderate, 3=severe.

    Returns:
        Noisy image.
    """
    noise_levels = {1: 0.05, 2: 0.1, 3: 0.2}
    noise_level = noise_levels.get(severity, 0.1)
    noise = torch.randn_like(image) * noise_level
    return torch.clamp(image + noise, 0, 1)


def darken_image(image: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    """Make image darker (simulates low lighting).

    Args:
        image: Image tensor (C, H, W).
        factor: 0.5 = 50% darker, 0.7 = 30% darker.

    Returns:
        Darkened image.
    """
    brightness_transform = transforms.ColorJitter(brightness=factor)
    return brightness_transform(image)


def test_robustness(model: torch.nn.Module, test_images: list[torch.Tensor]) -> dict:
    """Test model performance on original vs degraded images.

    Args:
        model: Your trained model.
        test_images: List of test images (clean).

    Returns:
        Dictionary with results showing accuracy drop.
    """
    logger.info("Starting robustness testing...")

    model.eval()
    device = next(model.parameters()).device

    # Test on clean images
    with torch.no_grad():
        clean_preds = []
        for img in test_images:
            img = img.unsqueeze(0).to(device)
            pred = model(img)
            clean_preds.append(pred)
        logger.info(f"Evaluated {len(clean_preds)} clean images")

    # Test on degraded images
    results = {}

    degradations = {
        "blur_mild": lambda img: blur_image(img, 1),
        "blur_severe": lambda img: blur_image(img, 3),
        "noise_mild": lambda img: add_noise(img, 1),
        "noise_severe": lambda img: add_noise(img, 3),
        "dark": lambda img: darken_image(img, 0.5),
    }

    for name, degrade_func in degradations.items():
        degraded_images = [degrade_func(img) for img in test_images]

        with torch.no_grad():
            degraded_preds = []
            for img in degraded_images:
                img = img.unsqueeze(0).to(device)
                pred = model(img)
                degraded_preds.append(pred)

        results[name] = {
            "sample_count": len(degraded_preds),
            "predictions": degraded_preds,
        }
        logger.info(f"Tested degradation: {name}")

    return {
        "clean_predictions": clean_preds,
        "degraded_results": results,
    }
