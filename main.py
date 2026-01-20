import io
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from ml_ops_project.model import Im2LatexModel
from ml_ops_project.preprocess import FormulaResizePad
from ml_ops_project.tokenizer import LaTeXTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/model.pth")
VOCAB_PATH = Path("models/vocab.pt")

model_artifacts: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model and tokenizer when the app starts.
    Clean up when the app stops.
    """
    # 1. LOAD TOKENIZER
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary not found at {VOCAB_PATH}")

    vocab = torch.load(VOCAB_PATH, map_location=DEVICE)
    tokenizer = LaTeXTokenizer()
    tokenizer.vocab = vocab
    tokenizer.idx_to_token = {v: k for k, v in vocab.items()}

    model_artifacts["tokenizer"] = tokenizer

    # 2. LOAD MODEL
    vocab_size = len(tokenizer.vocab)

    model = Im2LatexModel(
        vocab_size=vocab_size,
        d_model=64,
        nhead=4,
        num_decoder_layers=1,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    model_artifacts["model"] = model

    # 3. DEFINE TRANSFORM
    model_artifacts["transform"] = transforms.Compose(
        [
            FormulaResizePad(target_height=128, max_width=640),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    print(f"Model loaded on {DEVICE}")
    yield

    model_artifacts.clear()
    print("Model unloaded")


app = FastAPI(lifespan=lifespan)


def beam_search_prediction(model, image, tokenizer, beam_width=3, max_len=150):
    device = next(model.parameters()).device

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
def root():
    """Health check."""
    return {"message": "Im2Latex Inference API is running", "device": str(DEVICE), "status-code": HTTPStatus.OK}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Inference endpoint.
    Takes an image file, runs beam search, returns LaTeX string.
    """
    # 1. READ IMAGE
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file", "status-code": HTTPStatus.BAD_REQUEST}

    # 2. PREPROCESS
    transform = model_artifacts["transform"]
    input_tensor = transform(image)  # (3, 128, 640)

    # 3. INFERENCE
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]

    with torch.no_grad():
        prediction = beam_search_prediction(model, input_tensor, tokenizer)

    return {"filename": file.filename, "prediction": prediction, "status-code": HTTPStatus.OK}
