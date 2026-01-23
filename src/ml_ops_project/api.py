import io
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import storage  # type: ignore
from PIL import Image
from torchvision import transforms

from ml_ops_project.model import Im2LatexModel
from ml_ops_project.preprocess import FormulaResizePad
from ml_ops_project.tokenizer import LaTeXTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/model1.pth")
VOCAB_PATH = Path("models/vocab.pt")
LOG_FILE = Path("logs/prediction_database.csv")

model_artifacts: dict[str, Any] = {}


def _init_model_artifacts_if_needed() -> None:
    """Initialize model artifacts (tokenizer, model, transform)."""
    if {"tokenizer", "model", "transform"} <= model_artifacts.keys():
        return

    if VOCAB_PATH.exists():
        vocab = torch.load(VOCAB_PATH, map_location=DEVICE)
    else:
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2}

    tokenizer = LaTeXTokenizer()
    tokenizer.vocab = vocab
    tokenizer.idx_to_token = {v: k for k, v in vocab.items()}

    vocab_size = len(tokenizer.vocab)
    model = Im2LatexModel(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_decoder_layers=1,
    )

    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose(
        [
            FormulaResizePad(target_height=128, max_width=640),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    model_artifacts["tokenizer"] = tokenizer
    model_artifacts["model"] = model
    model_artifacts["transform"] = transform


def add_to_database(temp_img_path: Path, prediction: str) -> None:
    """Extracts features and appends them to the CSV log file."""
    try:
        # Extract features using your data_drift.py logic
        from ml_ops_project.data_drift import image_features

        features: dict[str, float | str] = {**image_features(temp_img_path)}

        # Add metadata and prediction
        features["prediction"] = prediction
        features["timestamp"] = datetime.now().isoformat()

        # Append to CSV
        df = pd.DataFrame([features])
        df.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)
    finally:
        # Clean up the temporary file
        if temp_img_path.exists():
            os.remove(temp_img_path)


def extract_image_features(image: Image.Image) -> dict[str, float]:
    """Compute simple image features directly from a PIL image (no temp files).

    Returns brightness, contrast, sharpness, width, height, aspect_ratio.
    """
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean())
    contrast = float(arr.std())
    gy, gx = np.gradient(arr)
    sharpness = float(np.abs(gx).mean() + np.abs(gy).mean())
    width, height = gray.size
    aspect_ratio = float(width / height) if height else 0.0
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": aspect_ratio,
    }


def save_prediction_to_gcp_record(features: dict[str, float], prediction: str) -> None:
    """Save a JSON record (features + prediction) to a GCS bucket."""
    record = {
        **features,
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
    }
    client = storage.Client()
    bucket_name = os.getenv("GCS_LOGGING_BUCKET", "ml_ops_data_bucket_46")
    bucket = client.bucket(bucket_name)
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    blob = bucket.blob(f"api_logs/{date_prefix}/prediction_{uuid4().hex}.json")
    blob.upload_from_string(json.dumps(record), content_type="application/json")


def save_prediction_locally_record(features: dict[str, float], prediction: str) -> None:
    """Save a JSONL record locally (no CSV)."""
    record = {
        **features,
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
    }
    out_dir = Path("logs/api_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    out_file = out_dir / f"predictions_{date_prefix}.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage app lifecycle: startup and shutdown."""
    if not LOG_FILE.exists():
        headers = ["brightness", "contrast", "sharpness", "width", "height", "aspect_ratio", "prediction", "timestamp"]
        pd.DataFrame(columns=headers).to_csv(LOG_FILE, index=False)
    try:
        _init_model_artifacts_if_needed()
    except Exception:
        pass
    yield
    model_artifacts.clear()


app = FastAPI(lifespan=lifespan)


def beam_search_prediction(
    model: torch.nn.Module,
    image: torch.Tensor,
    tokenizer: Any,
    beam_width: int = 3,
    max_len: int = 150,
) -> str:
    """Perform beam search decoding on image.

    Args:
        model: The encoder-decoder model.
        image: Input image tensor.
        tokenizer: LaTeX tokenizer.
        beam_width: Width of beam search.
        max_len: Maximum sequence length.

    Returns:
        LaTeX string prediction.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = DEVICE

    sos_id = tokenizer.vocab.get("<START>", tokenizer.vocab.get("<SOS>", 1))
    eos_id = tokenizer.vocab.get("<END>", tokenizer.vocab.get("<EOS>", 2))

    image = image.unsqueeze(0).to(device)

    start_seq = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    candidates = [(0.0, start_seq)]

    for _ in range(max_len):
        all_expansions = []
        for score, seq in candidates:
            if seq[0, -1].item() == eos_id:
                all_expansions.append((score, seq))
                continue

            output = model(image, seq)
            probs = torch.log_softmax(output[:, -1, :], dim=-1)
            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                new_seq = torch.cat([seq, topk_ids[0, i].view(1, 1)], dim=1)
                all_expansions.append((score + topk_probs[0, i].item(), new_seq))

        ordered = sorted(all_expansions, key=lambda x: x[0], reverse=True)
        candidates = ordered[:beam_width]

        if candidates[0][1][0, -1].item() == eos_id:
            break

    best_seq = candidates[0][1].squeeze().tolist()
    if isinstance(best_seq, int):
        best_seq = [best_seq]

    if len(best_seq) > 0 and best_seq[0] == sos_id:
        best_seq = best_seq[1:]
    if len(best_seq) > 0 and best_seq[-1] == eos_id:
        best_seq = best_seq[:-1]

    return " ".join([tokenizer.idx_to_token.get(t, "<UNK>") for t in best_seq])


@app.get("/")
def root() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Status information including device and health status.
    """
    return {
        "message": "Im2Latex Inference API is running",
        "device": str(DEVICE),
        "status-code": HTTPStatus.OK,
    }


@app.post("/predict/", response_model=None)
async def predict(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict[str, Any] | JSONResponse:
    """Inference endpoint that takes an image and returns LaTeX prediction.

    Args:
        file: Image file to process.

    Returns:
        JSON response with filename, prediction, and status code.

    Raises:
        HTTPException: If model artifacts not loaded or processing fails.
    """
    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=HTTPStatus.BAD_REQUEST, content={"error": "Empty file"})

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(status_code=HTTPStatus.BAD_REQUEST, content={"error": "Invalid image file"})

    _init_model_artifacts_if_needed()

    transform = model_artifacts.get("transform")
    model = model_artifacts.get("model")
    tokenizer = model_artifacts.get("tokenizer")
    if transform is None or model is None or tokenizer is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded",
        )

    try:
        input_tensor = transform(image)
        with torch.no_grad():
            prediction = beam_search_prediction(model, input_tensor, tokenizer)
    except Exception:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="Failed to process image",
        )

    # Extract features and log asynchronously (GCS JSON or local JSONL, no CSV)
    try:
        features = extract_image_features(image)
        if os.getenv("GCS_LOGGING_BUCKET"):
            background_tasks.add_task(save_prediction_to_gcp_record, features, prediction)
        else:
            background_tasks.add_task(save_prediction_locally_record, features, prediction)
    except Exception:
        # Do not fail the prediction if logging fails
        pass

    return {
        "filename": file.filename,
        "prediction": prediction,
        "status-code": HTTPStatus.OK,
    }
