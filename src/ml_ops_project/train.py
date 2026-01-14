"""Training script for Image-to-LaTeX model."""
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np

from ml_ops_project.model import Im2LatexModel
from ml_ops_project.data import MyDataset
from ml_ops_project.tokenizer import LaTeXTokenizer
from ml_ops_project.preprocess import get_train_transform, get_val_test_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def collate_fn(batch):
    """Collate function to handle variable-length sequences with padding.

    Args:
        batch: List of (image, label_tensor) tuples

    Returns:
        Tuple of (images, labels) where labels are padded to same length
    """
    images, labels = zip(*batch)
    images = torch.stack(images)
    
    # Pad labels to same length
    max_len = max(len(label) for label in labels)
    padded_labels = []
    for label in labels:
        padding = torch.zeros(max_len - len(label), dtype=torch.long)
        padded_labels.append(torch.cat([label, padding]))
    
    labels = torch.stack(padded_labels)
    return images, labels


def train(epochs: int = 10, batch_size: int = 32, data_path: str = "data/raw/default_train"):
    """Train the Image-to-LaTeX model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        data_path: Path to training data directory
    """
    data_path = Path(data_path)
    
    # Build tokenizer from labels
    print("Building tokenizer from labels...")
    tokenizer = LaTeXTokenizer()
    tokenizer.build_vocab(data_path / "labels.json")
    vocab_size = tokenizer.vocab_size
    pad_idx = tokenizer.get_pad_idx()
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets with appropriate transforms for each split
    print("Loading datasets...")
    train_transform = get_train_transform()
    val_test_transform = get_val_test_transform()
    
    # Create base dataset to get indices
    base_dataset = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=None,
    )
    
    # Split dataset: 70% train, 15% val, 15% test
    total_size = len(base_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Get indices from subsets
    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices
    
    # Create separate datasets with appropriate transforms
    train_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=train_transform,
    )
    train_dataset = torch.utils.data.Subset(train_dataset_base, train_indices)
    
    val_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=val_test_transform,
    )
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices)
    
    test_dataset_base = MyDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        transform=val_test_transform,
    )
    test_dataset = torch.utils.data.Subset(test_dataset_base, test_indices)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model = Im2LatexModel(
        vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_decoder_layers=3,
    )
    model.to(DEVICE)
    
    # Loss function (ignore padding tokens)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    statistics = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = []
        epoch_train_acc = []
        
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Teacher forcing: input is labels[:-1], target is labels[1:]
            # For each sequence, we predict the next token given previous tokens
            tgt_input = labels[:, :-1]  # Remove last token for input
            tgt_output = labels[:, 1:]  # Remove first token for target
            
            optimizer.zero_grad()
            y_pred = model(images, tgt_input)  # (Batch, Seq_Len-1, Vocab_Size)
            
            # Reshape for loss: (Batch * Seq_Len-1, Vocab_Size) and (Batch * Seq_Len-1,)
            loss = loss_fn(
                y_pred.reshape(-1, vocab_size),
                tgt_output.reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss.append(loss.item())
            
            # Calculate accuracy (excluding padding)
            pred_tokens = y_pred.argmax(dim=2)
            mask = (tgt_output != pad_idx)
            correct = (pred_tokens == tgt_output) * mask
            accuracy = correct.sum().float() / mask.sum().float()
            epoch_train_acc.append(accuracy.item())
            
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, acc: {accuracy.item():.4f}")
        
        # Validation phase
        model.eval()
        epoch_val_loss = []
        epoch_val_acc = []
        
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]
                
                y_pred = model(images, tgt_input)
                
                loss = loss_fn(
                    y_pred.reshape(-1, vocab_size),
                    tgt_output.reshape(-1)
                )
                
                epoch_val_loss.append(loss.item())
                
                pred_tokens = y_pred.argmax(dim=2)
                mask = (tgt_output != pad_idx)
                correct = (pred_tokens == tgt_output) * mask
                accuracy = correct.sum().float() / mask.sum().float()
                epoch_val_acc.append(accuracy.item())
        
        # Record statistics
        statistics["train_loss"].extend(epoch_train_loss)
        statistics["train_accuracy"].extend(epoch_train_acc)
        statistics["val_loss"].extend(epoch_val_loss)
        statistics["val_accuracy"].extend(epoch_val_acc)
        
        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_acc = np.mean(epoch_train_acc)
        avg_val_loss = np.mean(epoch_val_loss)
        avg_val_acc = np.mean(epoch_val_acc)
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        print()
    
    print("Training complete")
    
    # Save model and tokenizer
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    torch.save(tokenizer.vocab, "models/vocab.pt")
    print("Model and vocabulary saved to models/")
    
    # Save training statistics plots
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    axs[0, 0].plot(statistics["train_loss"])
    axs[0, 0].set_title("Train Loss")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Loss")
    
    axs[0, 1].plot(statistics["train_accuracy"])
    axs[0, 1].set_title("Train Accuracy")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Accuracy")
    
    axs[1, 0].plot(statistics["val_loss"])
    axs[1, 0].set_title("Validation Loss")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Loss")
    
    axs[1, 1].plot(statistics["val_accuracy"])
    axs[1, 1].set_title("Validation Accuracy")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Accuracy")
    
    plt.tight_layout()
    fig.savefig("reports/figures/training_statistics.png")
    print("Training statistics saved to reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
