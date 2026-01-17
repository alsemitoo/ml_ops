# tests/test_data.py
import json
import types
from pathlib import Path

import pytest
import torch
from ml_ops_project import data as data_module
from ml_ops_project.data import MyDataset
from ml_ops_project.tokenizer import LaTeXTokenizer
from PIL import Image


# -------- Fixtures --------
@pytest.fixture
def make_dataset_dir(tmp_path: Path):
    """Create a minimal dataset folder with one image and labels.json.

    Args:
        tmp_path: Pytest-provided temporary directory for test isolation.
    """

    def _make(
        *,
        text: str = r"\frac{1}{2}",
        image_name: str = "image_000000.png",
    ) -> Path:
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Save one image
        Image.new("RGB", (32, 32), (255, 255, 255)).save(images_dir / image_name)

        # Save labels.json
        labels = [{"image_file": image_name, "text": text}]
        (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

        return tmp_path

    return _make


@pytest.fixture
def make_empty_dir(tmp_path: Path):
    """Create dataset folder structure without labels.json to simulate empty data.

    Args:
        tmp_path: Pytest-provided temporary directory for test isolation.
    """

    def _make() -> Path:
        (tmp_path / "images").mkdir(parents=True, exist_ok=True)
        return tmp_path

    return _make


# -------- MyDataset tests --------
def test_dataset_with_labels_works(make_dataset_dir):
    """Happy path: dataset loads one sample and returns image plus torch.long labels.

    Args:
        make_dataset_dir: Fixture creating one-image dataset with labels.json.
    """
    base_path = make_dataset_dir()

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(base_path / "labels.json")

    dataset = MyDataset(base_path, tokenizer=tokenizer)

    assert len(dataset) == 1

    image, label_tensor = dataset[0]

    assert isinstance(image, Image.Image)
    assert isinstance(label_tensor, torch.Tensor)
    assert label_tensor.dtype == torch.long
    assert label_tensor.numel() > 0


def test_transform_is_applied(make_dataset_dir):
    """Transform is invoked and its output replaces the image from __getitem__.

    Args:
        make_dataset_dir: Fixture creating one-image dataset with labels.json.
    """
    base_path = make_dataset_dir()

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(base_path / "labels.json")

    called = {"value": False}

    def transform_fn(_img):
        called["value"] = True
        return "TRANSFORMED_IMAGE"

    # This is the correct runtime check
    assert callable(transform_fn)

    dataset = MyDataset(base_path, tokenizer=tokenizer, transform=transform_fn)

    image, _ = dataset[0]

    assert called["value"] is True
    assert image == "TRANSFORMED_IMAGE"


# -------- Error handling tests --------
def test_dataset_without_labels_is_empty(make_empty_dir):
    """Empty labels yield length zero and __getitem__ raises IndexError.

    Args:
        make_empty_dir: Fixture creating dataset directory without labels.json.
    """
    base_path = make_empty_dir()

    tokenizer = LaTeXTokenizer()
    dataset = MyDataset(base_path, tokenizer=tokenizer)

    assert len(dataset) == 0
    with pytest.raises(IndexError):
        _ = dataset[0]


def test_missing_image_raises_filenotfound(tmp_path: Path):
    """Missing image referenced in labels.json triggers FileNotFoundError on access.

    Args:
        tmp_path: Temporary directory used to write images and labels.json.
    """
    # Create labels.json but do NOT create the image file
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)

    labels = [{"image_file": "missing.png", "text": r"\frac{1}{2}"}]
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(tmp_path / "labels.json")

    dataset = MyDataset(tmp_path, tokenizer=tokenizer)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_transform_failure_is_propagated(make_dataset_dir):
    """Failing transform raises and propagates its exception from __getitem__.

    Args:
        make_dataset_dir: Fixture creating one-image dataset with labels.json.
    """
    base_path = make_dataset_dir()

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(base_path / "labels.json")

    def bad_transform(_img):
        raise ValueError("transform broke")

    dataset = MyDataset(base_path, tokenizer=tokenizer, transform=bad_transform)

    with pytest.raises(ValueError, match="transform broke"):
        _ = dataset[0]


def test_corrupted_image_file_raises(tmp_path: Path):
    """Corrupted image bytes cause PIL to raise when __getitem__ opens the file.

    Args:
        tmp_path: Temporary directory used to create corrupted image and labels.json.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Write garbage bytes instead of a real image
    (images_dir / "image_000000.png").write_bytes(b"not a real png")

    labels = [{"image_file": "image_000000.png", "text": r"\frac{1}{2}"}]
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(tmp_path / "labels.json")

    dataset = MyDataset(tmp_path, tokenizer=tokenizer)

    with pytest.raises(Exception):
        _ = dataset[0]


def test_malformed_labels_json_raises_json_decode_error(tmp_path: Path):
    """Invalid JSON in labels.json raises JSONDecodeError during dataset init.

    Args:
        tmp_path: Temporary directory used to create malformed labels.json.
    """
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)

    # Invalid JSON
    (tmp_path / "labels.json").write_text("{ this is not valid json", encoding="utf-8")

    tokenizer = LaTeXTokenizer()

    with pytest.raises(json.JSONDecodeError):
        _ = MyDataset(tmp_path, tokenizer=tokenizer)


def test_missing_keys_in_labels_raises_keyerror(tmp_path: Path):
    """Missing required keys in labels.json raises KeyError in __getitem__.

    Args:
        tmp_path: Temporary directory used to write labels.json missing keys.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # valid image exists, but labels are missing 'image_file' key
    Image.new("RGB", (32, 32), (255, 255, 255)).save(images_dir / "image_000000.png")

    labels = [{"text": r"\frac{1}{2}"}]  # missing "image_file"
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    tokenizer = LaTeXTokenizer()
    # build_vocab works because it reads item.get("text","")
    tokenizer.build_vocab(tmp_path / "labels.json")

    dataset = MyDataset(tmp_path, tokenizer=tokenizer)

    with pytest.raises(KeyError):
        _ = dataset[0]


# -------- download_data tests --------
def test_download_data_saves_images_and_labels(tmp_path, monkeypatch):
    """download_data saves images and labels.json when load_dataset returns one sample.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to stub load_dataset.
    """

    # Fake dataset with one sample
    class FakeHFDataset(list):
        pass

    fake_dataset = FakeHFDataset(
        [
            {"image": Image.new("RGB", (10, 10), (255, 255, 255)), "text": r"\frac{1}{2}"},
        ]
    )

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="default",
            split="train",
            output_path=str(tmp_path),
        )
    )

    # call download_data
    data_module.download_data.__wrapped__(cfg)

    dataset_folder = tmp_path / "default_train"
    image_path = dataset_folder / "images" / "image_000000.png"
    labels_path = dataset_folder / "labels.json"

    assert image_path.exists()
    assert labels_path.exists()

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    assert labels == [{"image_file": "image_000000.png", "text": r"\frac{1}{2}"}]


def test_progress_log_every_100(tmp_path, monkeypatch):
    """Progress log emits at 100 samples processed.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to stub load_dataset and logger.info.
    """
    # Fake dataset with 100 samples
    fake_dataset = []
    for _ in range(100):
        fake_dataset.append(
            {
                "image": Image.new("RGB", (5, 5), (255, 255, 255)),
                "text": "a",
            }
        )

    def fake_load_dataset(*args, **kwargs):
        return fake_dataset

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    # capture logger info calls
    calls = []

    def fake_logger_info(msg):
        calls.append(msg)

    monkeypatch.setattr(data_module.logger, "info", fake_logger_info)

    # minimal cfg
    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="small",
            split="train",
            output_path=str(tmp_path),
        )
    )
    # call download_data
    data_module.download_data.__wrapped__(cfg)

    assert any("Processed 100/100" in msg for msg in calls)


# -------- Error handling tests for download_data --------
def test_download_data_raises_if_load_dataset_fails(tmp_path: Path, monkeypatch):
    """Exceptions from load_dataset propagate out of download_data.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to force load_dataset failure.
    """

    def fake_load_dataset(*args, **kwargs):
        raise ValueError("Dataset not found")

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="does_not_exist",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(ValueError, match="Dataset not found"):
        data_module.download_data.__wrapped__(cfg)


def test_download_data_fails_if_item_missing_image(tmp_path: Path, monkeypatch):
    """Dataset item missing 'image' key raises KeyError during save.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to provide malformed dataset.
    """
    fake_dataset = [{"text": r"\frac{1}{2}"}]  # missing "image"

    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: fake_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="default",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(KeyError):
        data_module.download_data.__wrapped__(cfg)


def test_download_data_fails_if_image_has_no_save(tmp_path: Path, monkeypatch):
    """Dataset item image object without .save raises AttributeError.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to provide malformed dataset.
    """
    fake_dataset = [{"image": object(), "text": "a"}]  # object() has no .save()

    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: fake_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="default",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(AttributeError):
        data_module.download_data.__wrapped__(cfg)


def test_download_data_load_dataset_failure_is_propagated(tmp_path: Path, monkeypatch):
    """Runtime errors from load_dataset propagate out of download_data.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to force load_dataset runtime error.
    """

    def fake_load_dataset(*args, **kwargs):
        raise RuntimeError("hf download failed")

    monkeypatch.setattr(data_module, "load_dataset", fake_load_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="small",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(RuntimeError, match="hf download failed"):
        data_module.download_data.__wrapped__(cfg)


def test_download_data_item_missing_image_key_raises_keyerror(tmp_path: Path, monkeypatch):
    """Duplicate coverage: missing image key in dataset raises KeyError.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to provide malformed dataset.
    """
    fake_dataset = [{"text": "a"}]  # missing "image"

    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: fake_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="small",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(KeyError):
        data_module.download_data.__wrapped__(cfg)


def test_download_data_image_object_without_save_raises_attributeerror(tmp_path: Path, monkeypatch):
    """Duplicate coverage: image object lacking .save triggers AttributeError.

    Args:
        tmp_path: Temporary output directory.
        monkeypatch: Pytest monkeypatch fixture to provide malformed dataset.
    """
    fake_dataset = [{"image": object(), "text": "a"}]  # object() has no .save()

    monkeypatch.setattr(data_module, "load_dataset", lambda *a, **k: fake_dataset)

    cfg = types.SimpleNamespace(
        download=types.SimpleNamespace(
            name="small",
            split="train",
            output_path=str(tmp_path),
        )
    )

    with pytest.raises(AttributeError):
        data_module.download_data.__wrapped__(cfg)
