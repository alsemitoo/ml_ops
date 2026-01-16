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

        # --- 1. RESNET SURGERY (High Resolution Encoder) ---
        # Same as Fix 2: We keep strides=1 in layer 3 and 4
        resnet = models.resnet18(weights="DEFAULT")

        # Modify strides to reduce downsampling from 32x -> 8x
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)

        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_adapter = nn.Conv2d(512, d_model, kernel_size=1)

        # --- 2. 2D POSITIONAL ENCODING (The Upgrade) ---
        # We replace the 1D massive list with two smaller lookup tables.
        # d_model is split: 128 for Y (height), 128 for X (width).

        # We define "safe" maximum dimensions for the feature map.
        # With 8x downsampling:
        # Max Height 128px -> 16 feats. We set limit to 50.
        # Max Width 1000px -> 125 feats. We set limit to 300.
        self.max_h = 50
        self.max_w = 300

        self.row_embed = nn.Embedding(self.max_h, d_model // 2)
        self.col_embed = nn.Embedding(self.max_w, d_model // 2)

        # --- 3. DECODER ---
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def get_2d_pos_encoding(self, h, w, device):
        """
        Generates the 2D embedding grid on the fly.
        """
        # 1. Create the grid coordinates
        # y_indices: [[0, 0, ..., 0], [1, 1, ..., 1], ...]
        # x_indices: [[0, 1, ..., w], [0, 1, ..., w], ...]
        y_range = torch.arange(h, device=device)
        x_range = torch.arange(w, device=device)

        # meshgrid 'ij' indexing creates the grid we need
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        # 2. Look up embeddings
        # shape: (H, W, d_model/2)
        y_enc = self.row_embed(y_grid)
        x_enc = self.col_embed(x_grid)

        # 3. Concatenate to get full d_model
        # shape: (H, W, d_model)
        pos_enc = torch.cat([y_enc, x_enc], dim=-1)

        # 4. Flatten and permute to match Transformer format
        # (H, W, d_model) -> (H*W, d_model) -> (H*W, 1, d_model)
        # We add the '1' dimension so it broadcasts across the batch
        return pos_enc.flatten(0, 1).unsqueeze(1)

    def forward(self, images, tgt_text, tgt_pad_mask=None):
        """
        images: (Batch, 3, H, W)
        """

        # --- ENCODER ---
        features = self.backbone(images)
        features = self.feature_adapter(features)  # (Batch, 256, H_feat, W_feat)

        B, C, H, W = features.shape

        # Flatten features: (Seq_Len, Batch, d_model)
        memory = features.flatten(2).permute(2, 0, 1)

        # --- APPLY 2D POS ENCODING ---
        # We generate the encoding based on the CURRENT feature map size
        # giving us flexibility if image sizes vary slightly.
        if H > self.max_h or W > self.max_w:
            print(f"Warning: Image feature map ({H}x{W}) exceeds pos encoding limits ({self.max_h}x{self.max_w})")

        pos_enc = self.get_2d_pos_encoding(H, W, images.device)

        # Add to memory
        memory = memory + pos_enc

        # --- DECODER ---
        tgt_emb = self.embedding(tgt_text).permute(1, 0, 2)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(images.device)

        output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)

        prediction = self.fc_out(output)
        return prediction.permute(1, 0, 2)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    # Test
    model = Im2LatexModel(vocab_size=100)
    # Batch of 128x640 images
    x = torch.rand(2, 3, 128, 640)
    tgt = torch.randint(0, 100, (2, 20))

    out = model(x, tgt)
    print(f"Output shape: {out.shape}")  # Should be (2, 20, 100)
