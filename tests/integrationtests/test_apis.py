from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from ml_ops_project.api import app, model_artifacts

client = TestClient(app)


def test_read_root() -> None:
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Im2Latex Inference API is running"
    assert "device" in data
    assert data["status-code"] == 200


def create_test_image() -> BytesIO:
    """Create a simple test image."""
    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return img_byte_arr


def test_predict_endpoint_with_valid_image() -> None:
    """Test prediction endpoint with valid image."""
    test_image = create_test_image()
    response = client.post("/predict/", files={"file": ("test.png", test_image, "image/png")})

    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "prediction" in data
    assert "status-code" in data
    assert data["filename"] == "test.png"
    assert data["status-code"] == 200


def test_predict_endpoint_with_invalid_file() -> None:
    """Test prediction endpoint with invalid image file."""
    response = client.post("/predict/", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "error" in response.json()


def test_predict_endpoint_with_empty_file() -> None:
    """Test prediction endpoint with empty file."""
    response = client.post("/predict/", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "Empty file"


def test_predict_endpoint_with_corrupted_image() -> None:
    """Test prediction endpoint with corrupted image data."""
    corrupted_data = b"\x89PNG\r\n\x1a\n" + b"corrupted"
    response = client.post("/predict/", files={"file": ("corrupt.png", corrupted_data, "image/png")})
    assert response.status_code == 400
    assert "error" in response.json()


def test_model_artifacts_already_loaded() -> None:
    """Test that model artifacts persist across multiple requests."""
    test_image = create_test_image()
    response = client.post("/predict/", files={"file": ("test.png", test_image, "image/png")})
    assert response.status_code == 200
