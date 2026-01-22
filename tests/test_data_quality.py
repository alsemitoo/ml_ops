import json
from pathlib import Path

import pytest
from PIL import Image


# Data validation tests (run on actual data)
@pytest.fixture
def real_data_path() -> Path:
    """Path to actual data pulled via DVC."""
    data_path = Path("data/raw/default_train")
    if not data_path.exists():
        pytest.skip("Data not available")
    return data_path


@pytest.fixture
def sample_size(request) -> tuple[int, int]:
    """Return sample range based on test markers."""
    if request.node.get_closest_marker("full_dataset"):
        # Return full dataset range
        labels_file = Path("data/raw/default_train/labels.json")
        if labels_file.exists():
            labels = json.loads(labels_file.read_text(encoding="utf-8"))
            return (0, len(labels))
    # Return default sample range
    return (0, 1000)


def test_labels_json_exists(real_data_path: Path) -> None:
    """Verify labels.json exists in the data directory."""
    labels_file = real_data_path / "labels.json"
    assert labels_file.exists(), "labels.json not found in data directory"


def test_labels_json_is_valid(real_data_path: Path) -> None:
    """Verify labels.json is valid JSON with expected structure."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))

    assert isinstance(labels, list), "labels should be a list"
    assert len(labels) > 0, "labels should not be empty"

    # Check first entry has required keys
    assert "image_file" in labels[0], "Missing 'image_file' key"
    assert "text" in labels[0], "Missing 'text' key"


def test_referenced_images_exist(real_data_path: Path) -> None:
    """Verify a sample of images referenced in labels.json actually exist."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    missing_images = []
    for item in labels[50000:51000]:
        image_path = images_dir / item["image_file"]
        if not image_path.exists():
            missing_images.append(item["image_file"])

    assert len(missing_images) == 0, f"Missing images: {missing_images}"


@pytest.mark.full_dataset
def test_all_referenced_images_exist(real_data_path: Path) -> None:
    """Verify ALL images referenced in labels.json exist (run with -m full_dataset)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    missing_images = []
    for i, item in enumerate(labels):
        if i % 10000 == 0:
            print(f"Checking existence {i}/{len(labels)}...")

        image_path = images_dir / item["image_file"]
        if not image_path.exists():
            missing_images.append(item["image_file"])

    assert len(missing_images) == 0, f"Found {len(missing_images)} missing images: {missing_images[:10]}"


def test_images_are_loadable(real_data_path: Path) -> None:
    """Verify a sample of images can be opened by PIL."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    for item in labels[2000:3000]:
        image_path = images_dir / item["image_file"]
        try:
            img = Image.open(image_path)
            img.verify()  # Verify it's a valid image
        except Exception as e:
            pytest.fail(f"Failed to load {item['image_file']}: {e}")


@pytest.mark.full_dataset
def test_all_images_are_loadable(real_data_path: Path) -> None:
    """Verify ALL images can be opened by PIL (run with -m full_dataset)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    corrupted_images = []
    for i, item in enumerate(labels):
        if i % 10000 == 0:
            print(f"Loading image {i}/{len(labels)}...")

        image_path = images_dir / item["image_file"]
        try:
            img = Image.open(image_path)
            img.verify()
        except Exception as e:
            corrupted_images.append((item["image_file"], str(e)))

    assert len(corrupted_images) == 0, f"Found {len(corrupted_images)} corrupted images: {corrupted_images[:10]}"


def test_data_size_is_reasonable(real_data_path: Path) -> None:
    """Verify dataset has expected number of samples."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))

    # Adjust these bounds based on your expected dataset size
    assert len(labels) > 50000, "Dataset too small"
    assert len(labels) < 100000, "Dataset unexpectedly large"


def test_images_have_reasonable_dimensions(real_data_path: Path) -> None:
    """Verify a sample of images have reasonable dimensions (not too small/large)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    for item in labels[:1000]:
        image_path = images_dir / item["image_file"]
        img = Image.open(image_path)
        width, height = img.size

        assert width > 10, f"{item['image_file']}: width too small ({width})"
        assert height > 10, f"{item['image_file']}: height too small ({height})"
        assert width < 5000, f"{item['image_file']}: width too large ({width})"
        assert height < 5000, f"{item['image_file']}: height too large ({height})"


@pytest.mark.full_dataset
def test_all_images_have_reasonable_dimensions(real_data_path: Path) -> None:
    """Verify ALL images have reasonable dimensions (run with -m full_dataset)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    bad_dimensions = []
    for i, item in enumerate(labels):
        if i % 10000 == 0:
            print(f"Checking dimensions {i}/{len(labels)}...")

        image_path = images_dir / item["image_file"]
        img = Image.open(image_path)
        width, height = img.size

        if width <= 10 or height <= 10 or width >= 5000 or height >= 5000:
            bad_dimensions.append((item["image_file"], width, height))

    assert len(bad_dimensions) == 0, f"Found {len(bad_dimensions)} images with bad dimensions: {bad_dimensions[:10]}"


def test_images_are_not_blank(real_data_path: Path) -> None:
    """Verify sample of images contain actual content (not completely white/empty)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    for item in labels[23000:24000]:
        image_path = images_dir / item["image_file"]
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        pixels: list[int] = list(img.get_flattened_data())  # type: ignore[arg-type]
        avg_brightness: float = sum(pixels) / len(pixels)

        # Check for completely blank images (all pixels same value)
        # For white bg with small black formula, avg will be ~250-253
        assert avg_brightness < 254, f"{item['image_file']}: appears to be blank/all white (avg: {avg_brightness:.2f})"
        # If average is close to 0 (black), image is likely corrupted
        assert avg_brightness > 1, f"{item['image_file']}: appears to be all black (avg: {avg_brightness:.2f})"


@pytest.mark.full_dataset
def test_all_images_are_not_blank(real_data_path: Path) -> None:
    """Verify ALL images contain actual content (run with -m full_dataset)."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    blank_images = []
    for i, item in enumerate(labels):
        if i % 10000 == 0:
            print(f"Checking image {i}/{len(labels)}...")

        image_path = images_dir / item["image_file"]
        img = Image.open(image_path).convert("L")
        pixels: list[int] = list(img.get_flattened_data())  # type: ignore[arg-type]
        avg_brightness: float = sum(pixels) / len(pixels)

        if avg_brightness >= 254.9:
            blank_images.append((item["image_file"], avg_brightness))

    assert len(blank_images) == 0, f"Found {len(blank_images)} blank images: {blank_images[:10]}"


def test_image_mode_is_consistent(real_data_path: Path) -> None:
    """Verify sample of images have consistent color mode."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    modes = set()
    for item in labels[37000:38000]:
        image_path = images_dir / item["image_file"]
        img = Image.open(image_path)
        modes.add(img.mode)

    # Allow RGB, grayscale, or binary
    allowed_modes = {"RGB", "L", "1", "LA", "RGBA"}
    assert modes.issubset(allowed_modes), f"Unexpected image modes found: {modes - allowed_modes}"


@pytest.mark.full_dataset
def test_all_images_mode_is_consistent(real_data_path: Path) -> None:
    """Verify sample of images have consistent color mode."""
    labels_file = real_data_path / "labels.json"
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    images_dir = real_data_path / "images"

    modes = set()
    for i, item in enumerate(labels):
        if i % 10000 == 0:
            print(f"Checking image mode {i}/{len(labels)}...")

        image_path = images_dir / item["image_file"]
        img = Image.open(image_path)
        modes.add(img.mode)

    # Allow RGB, grayscale, or binary
    allowed_modes = {"RGB", "L", "1", "LA", "RGBA"}
    assert modes.issubset(allowed_modes), f"Unexpected image modes found: {modes - allowed_modes}"
