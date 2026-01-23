import pytest
from PIL import Image

from ml_ops_project.preprocess import FormulaResizePad


def test_formula_resize_pad_output_size_and_padding():
    """Verify __call__ resizes to target height and pads to max width with white background."""
    # Create a solid red image (width=50, height=50)
    img = Image.new("RGB", (50, 50), (255, 0, 0))

    transform = FormulaResizePad(target_height=100, max_width=150)
    out = transform(img)

    # Output size should be (max_width, target_height)
    assert out.size == (150, 100)

    # Left region (within resized content) should contain red pixel
    assert out.getpixel((10, 10)) == (255, 0, 0)

    # Right padded region should be white
    assert out.getpixel((149, 50)) == (255, 255, 255)


@pytest.mark.xfail(strict=True, reason="Demonstration: intentionally incorrect expectation")
def test_formula_resize_pad_expected_fail_wrong_size():
    """Intentionally failing test: expects wrong output size."""
    img = Image.new("RGB", (80, 40), (0, 255, 0))
    transform = FormulaResizePad(target_height=100, max_width=200)
    out = transform(img)

    # Wrong expectation: actual should be (200, 100)
    assert out.size == (201, 100)


def test_formula_resize_pad_clamps_when_exceeds_max_width():
    """When scaled width exceeds max_width, output fills full width (no white padding)."""
    # Construct an image that will greatly exceed max_width after scaling
    img = Image.new("RGB", (2000, 10), (0, 0, 255))

    transform = FormulaResizePad(target_height=100, max_width=120)
    out = transform(img)

    # Should clamp to max_width
    assert out.size == (120, 100)
    # Far-right pixel should be blue (content), not white padding
    assert out.getpixel((119, 50)) == (0, 0, 255)
