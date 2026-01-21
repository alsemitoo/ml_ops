"""Simple tests for model robustness under image degradation.

Tests verify the model still works when images are blurry, noisy, or dark.
"""

import pytest
import torch

from ml_ops_project.drift_detection import add_noise, blur_image, darken_image, test_robustness


class TestImageDegradations:
    """Test that image degradation functions work correctly."""

    @pytest.fixture
    def sample_image(self) -> torch.Tensor:
        """Create a test image."""
        return torch.rand(3, 32, 32)

    def test_blur_works(self, sample_image: torch.Tensor) -> None:
        """Test blur function."""
        blurred = blur_image(sample_image, severity=2)
        assert blurred.shape == sample_image.shape
        assert not torch.allclose(blurred, sample_image)

    def test_noise_works(self, sample_image: torch.Tensor) -> None:
        """Test noise function."""
        noisy = add_noise(sample_image, severity=1)
        assert noisy.shape == sample_image.shape
        assert noisy.min() >= 0 and noisy.max() <= 1

    def test_darken_works(self, sample_image: torch.Tensor) -> None:
        """Test darkening function."""
        dark = darken_image(sample_image, factor=0.5)
        assert dark.shape == sample_image.shape


class TestRobustnessTesting:
    """Test robustness evaluation on a simple model."""

    @pytest.fixture
    def simple_model(self) -> torch.nn.Module:
        """Create a basic test model."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 5),
        )

    def test_robustness_function(self, simple_model: torch.nn.Module) -> None:
        """Test the main robustness testing function."""
        # Create 3 test images
        test_images = [torch.rand(3, 32, 32) for _ in range(3)]

        # Run robustness test
        results = test_robustness(simple_model, test_images)

        # Check we got results for clean and all degraded versions
        assert "clean_predictions" in results
        assert "degraded_results" in results
        assert len(results["clean_predictions"]) == 3

        # Check we tested all degradation types
        degradation_types = ["blur_mild", "blur_severe", "noise_mild", "noise_severe", "dark"]
        for deg_type in degradation_types:
            assert deg_type in results["degraded_results"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
