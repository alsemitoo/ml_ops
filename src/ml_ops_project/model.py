"""Image-to-LaTeX model using ResNet encoder and Transformer decoder."""
import torch
import torch.nn as nn
import torchvision.models as models
from loguru import logger


class Im2LatexModel(nn.Module):
    """Image-to-LaTeX sequence-to-sequence model.

    Architecture:
        - Encoder: ResNet-18 for visual feature extraction
        - Decoder: Transformer decoder for LaTeX token generation
        - Positional encoding: Learnable position embeddings for image patches
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_decoder_layers: int = 3,
    ) -> None:
        """Initialize the Image-to-LaTeX model.

        Args:
            vocab_size: Size of the LaTeX token vocabulary
            d_model: Dimensionality of embeddings and hidden states (default: 256)
            nhead: Number of attention heads in transformer (default: 4)
            num_decoder_layers: Number of transformer decoder layers (default: 3)
        """
        super().__init__()
        logger.info(
            f"Initializing Im2LatexModel with vocab_size={vocab_size}, "
            f"d_model={d_model}, nhead={nhead}, num_decoder_layers={num_decoder_layers}"
        )

        # --- 1. ENCODER: ResNet-18 for visual feature extraction ---
        # Remove last 3 layers to preserve spatial dimensions (H', W')
        resnet = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        # Adapt ResNet output channels (256) to d_model
        self.feature_adapter = nn.Conv2d(256, d_model, kernel_size=1)

        # --- 2. POSITIONAL ENCODING ---
        # Learnable position embeddings for flattened image patches
        # Assuming max image compression: 128x640 -> 8x40 = 320 patches
        self.pos_encoder = nn.Parameter(torch.randn(320, 1, d_model))

        # --- 3. DECODER: Transformer for sequence generation ---
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.parameters())} "
            "total parameters"
        )

    def forward(
        self,
        images: torch.Tensor,
        tgt_text: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through encoder and decoder.

        Args:
            images: Batch of images with shape (Batch, 3, Height, Width)
            tgt_text: Batch of LaTeX token sequences with shape (Batch, Seq_Len)

        Returns:
            Predictions with shape (Batch, Seq_Len, Vocab_Size)
        """
        # --- ENCODER STEP ---
        # 1. Extract visual features with ResNet backbone
        features = self.backbone(images)  # (Batch, 256, H', W')

        # 2. Adapt feature dimensions
        features = self.feature_adapter(features)  # (Batch, d_model, H', W')

        # 3. Flatten and permute for transformer (sequence-first format)
        batch_size, channels, height, width = features.shape
        memory = features.flatten(2).permute(2, 0, 1)  # (Seq_Len, Batch, d_model)

        # 4. Add positional encoding to memory
        seq_len = memory.size(0)
        pos_enc = self.pos_encoder[:seq_len, :, :].squeeze(1)  # (Seq_Len, d_model)
        memory = memory + pos_enc.unsqueeze(1)  # Broadcast to (Seq_Len, Batch, d_model)

        # --- DECODER STEP ---
        # 1. Embed target text sequences
        tgt_emb = self.embedding(tgt_text).permute(1, 0, 2)  # (Seq_Len, Batch, d_model)

        # 2. Generate causal mask to prevent attending to future tokens
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0)).to(
            images.device
        )

        # 3. Run transformer decoder
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
        )

        # 4. Project to vocabulary space
        prediction = self.fc_out(output)  # (Seq_Len, Batch, Vocab_Size)

        # Return in batch-first format for compatibility
        return prediction.permute(1, 0, 2)

    @staticmethod
    def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate causal attention mask for autoregressive decoding.

        Args:
            sz: Size of the mask (sequence length)

        Returns:
            Causal mask tensor of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0)
        )
        return mask


if __name__ == "__main__":
    logger.info("Testing Im2LatexModel with dummy inputs...")
    model = Im2LatexModel(vocab_size=100)
    x = torch.rand(1, 3, 224, 224)
    tgt_text = torch.randint(0, 100, (1, 10))
    output = model(x, tgt_text)
    logger.success(f"Output shape: {output.shape} (expected: (1, 10, 100))")
