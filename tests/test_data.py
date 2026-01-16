import json
from pathlib import Path

from ml_ops_project.data import MyDataset
from ml_ops_project.tokenizer import LaTeXTokenizer
from PIL import Image
from torch.utils.data import Dataset

def test_my_dataset(tmp_path: Path) -> None:
    """Test the MyDataset class."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / "image_000000.png"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(image_path)

    labels = [{"image_file": "image_000000.png", "text": r"\frac{1}{2}"}]
    (tmp_path / "labels.json").write_text(json.dumps(labels), encoding="utf-8")

    tokenizer = LaTeXTokenizer()
    dataset = MyDataset(tmp_path, tokenizer=tokenizer)

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
