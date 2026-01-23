from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


class APIUser(HttpUser):
    """Simulate users making requests to the API."""

    wait_time = between(1, 3)

    def on_start(self):
        """Called when a simulated user starts."""
        self.client.verify = False

    @task(1)
    def test_health_check(self):
        """Test the health check endpoint."""
        self.client.get("/")

    @task(3)
    def test_predict_with_valid_image(self):
        """Test prediction with a valid image."""
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        self.client.post(
            "/predict/",
            files={"file": ("test.png", img_bytes.getvalue(), "image/png")},
        )

    @task(1)
    def test_predict_with_invalid_image(self):
        with self.client.post(
            "/predict/",
            files={"file": ("test.txt", b"invalid", "text/plain")},
            catch_response=True,
            name="POST /predict/ (invalid)",
        ) as resp:
            if resp.status_code == 400:
                resp.success()
            else:
                resp.failure(f"Expected 400, got {resp.status_code}: {resp.text}")
