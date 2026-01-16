import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from hydra import main as hydra_main
from omegaconf import DictConfig
from loguru import logger
from torch.amp import autocast
from tqdm import tqdm

from ml_ops_project.data import MyDataset
from ml_ops_project.model import Im2LatexModel
from ml_ops_project.preprocess import get_val_test_transform
from ml_ops_project.tokenizer import LaTeXTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    max_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        padding = torch.zeros(max_len - len(label), dtype=torch.long)
        padded_labels.append(torch.cat([label, padding]))
    return images, torch.stack(padded_labels)


def get_test_dataset(data_path: Path, tokenizer: LaTeXTokenizer):
    transform = get_val_test_transform()
    base_dataset = MyDataset(data_path=data_path, tokenizer=tokenizer, transform=transform)
    total_size = len(base_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    _, _, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    return test_subset


# --- FIXED GENERATION FUNCTION ---
import torch
import math


def beam_search_prediction(model, image, tokenizer, beam_width=3, max_len=150):
    """
    Predicts using Beam Search to reduce repetition and hallucinations.
    """
    model.eval()

    # FIX 1: Get device from model parameters
    device = next(model.parameters()).device

    # Setup IDs
    sos_id = tokenizer.token_to_id.get("<START>", tokenizer.token_to_id.get("<SOS>"))
    eos_id = tokenizer.token_to_id.get("<END>", tokenizer.token_to_id.get("<EOS>"))

    # Fallback
    if sos_id is None:
        sos_id = 1
    if eos_id is None:
        eos_id = 2

    # Prepare Image
    image = image.unsqueeze(0).to(device)  # (1, 3, H, W)
    curr_img = image.repeat(beam_width, 1, 1, 1)  # (K, 3, H, W)

    # Initialize Beams
    start_seq = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    candidates = [(0.0, start_seq)]

    # Beam Search Loop
    for step in range(max_len):
        all_expansions = []

        for score, seq in candidates:
            # If beam finished, keep it
            if seq[0, -1].item() == eos_id:
                all_expansions.append((score, seq))
                continue

            # Forward Pass
            # seq: (1, Seq_Len)
            output = model(image, seq)

            # Log Softmax
            probs = torch.log_softmax(output[:, -1, :], dim=-1)

            # Top K
            topk_probs, topk_ids = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                next_score = topk_probs[0, i].item()
                next_id = topk_ids[0, i].view(1, 1)

                new_seq = torch.cat([seq, next_id], dim=1)
                new_score = score + next_score
                all_expansions.append((new_score, new_seq))

        # Select best candidates
        ordered = sorted(all_expansions, key=lambda x: x[0], reverse=True)
        candidates = ordered[:beam_width]

        # Stop if best candidate is EOS
        if candidates[0][1][0, -1].item() == eos_id:
            break

    # Get Winner
    best_seq = candidates[0][1]
    tokens = best_seq.squeeze().tolist()

    # Handle single token edge case
    if isinstance(tokens, int):
        tokens = [tokens]

    # DEBUG: Print Raw IDs to console (so you can see if model is predicting ANYTHING)
    # This will appear in your logs during generation
    print(f"DEBUG: Best Beam Raw IDs: {tokens}")

    # FIX 2: Clean up logic
    if len(tokens) > 0 and tokens[0] == sos_id:
        tokens = tokens[1:]
    if len(tokens) > 0 and tokens[-1] == eos_id:
        tokens = tokens[:-1]

    # FIX 3: MANUAL DECODE (Bypass tokenizer.decode)
    # We use the id_to_token map directly.
    decoded_strings = [tokenizer.id_to_token.get(t, "<UNK>") for t in tokens]

    return decoded_strings  # Returns list of strings e.g. ['\hat', '{', 'a', '}']


def evaluate_metrics(model, dataloader, loss_fn, vocab_size, pad_idx):
    model.eval()
    total_loss = 0
    total_acc = 0
    steps = 0

    logger.info("Running quantitative evaluation...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                y_pred = model(images, tgt_input)
                loss = loss_fn(y_pred.reshape(-1, vocab_size), tgt_output.reshape(-1))

            total_loss += loss.item()
            pred_tokens = y_pred.argmax(dim=2)
            mask = tgt_output != pad_idx
            correct = (pred_tokens == tgt_output) * mask
            accuracy = correct.sum().float() / mask.sum().float()
            total_acc += accuracy.item()
            steps += 1

    return total_loss / steps, total_acc / steps


@hydra_main(config_path="../../configs", config_name="train", version_base=None)
def test(cfg: DictConfig):
    test_cfg = cfg.training
    model_cfg = cfg.model
    data_path = Path(test_cfg.data_path)

    logger.info(f"Running testing on device: {DEVICE}")

    # 1. LOAD VOCAB
    vocab_path = Path("models/vocab.pt")
    if not vocab_path.exists():
        logger.error("Vocabulary missing!")
        return

    vocab = torch.load(vocab_path)
    # Manual Inversion to ensure we have ID -> Token
    id_to_token = {v: k for k, v in vocab.items()}

    # Initialize Tokenizer
    tokenizer = LaTeXTokenizer()
    tokenizer.vocab = vocab
    tokenizer.token_to_id = vocab
    tokenizer.id_to_token = id_to_token

    pad_idx = tokenizer.get_pad_idx()
    vocab_size = len(tokenizer.vocab)

    # 2. FIND SPECIAL TOKENS
    possible_sos = ["<START>", "<SOS>", "<s>", "<start>", "[SOS]", "[START]"]
    possible_eos = ["<END>", "<EOS>", "</s>", "<end>", "[EOS]", "[END]"]

    sos_id = None
    eos_id = None

    for token in possible_sos:
        if token in tokenizer.token_to_id:
            sos_id = tokenizer.token_to_id[token]
            break
    for token in possible_eos:
        if token in tokenizer.token_to_id:
            eos_id = tokenizer.token_to_id[token]
            break

    # 3. LOAD DATA & MODEL
    test_dataset = get_test_dataset(data_path, tokenizer)
    # Shuffle=True just to see different random samples every run
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    model = Im2LatexModel(
        vocab_size=vocab_size,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_decoder_layers=model_cfg.num_decoder_layers,
    )
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # 4. METRICS
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    avg_loss, avg_acc = evaluate_metrics(model, test_dataloader, loss_fn, vocab_size, pad_idx)
    logger.info(f"TEST RESULTS | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    # 5. DEBUG GENERATION
    logger.info("Generating predictions for 3 random test samples...")
    indices = torch.randperm(len(test_dataset))[:3]

    for idx in indices:
        image, label_ids = test_dataset[idx]

        # Ground Truth Manual Decode
        gt_tokens = [id_to_token.get(i, "<UNK>") for i in label_ids.tolist()]
        gt_str = " ".join([t for t in gt_tokens if t not in ["<PAD>", "<START>", "<END>"]])

        # Prediction Manual Decode
        # Pass id_to_token explicitly!
        pred_tokens = beam_search_prediction(model, image, tokenizer)
        pred_str = " ".join(pred_tokens)

        print("-" * 50)
        print(f"Sample ID: {idx.item()}")
        print(f"GT IDs:       {label_ids.tolist()[:10]} ...")
        print(f"Ground Truth: {gt_str}")
        print(f"Prediction:   {pred_str}")
        print("-" * 50)


if __name__ == "__main__":
    test()
