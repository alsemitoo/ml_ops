import torch
import torch.nn as nn
import torchvision.models as models


class Im2LatexModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_decoder_layers=3):
        super().__init__()

        # --- 1. SIMPLIFIED ENCODER (Replaces Swin) ---
        # We use a standard ResNet-18 to extract visual features.
        # We remove the last two layers (avgpool and fc) to keep spatial features.
        resnet = models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        # This 1x1 convolution adapts the ResNet output channels (512) to our d_model (256)
        self.feature_adapter = nn.Conv2d(256, d_model, kernel_size=1)

        # --- 2. POSITIONAL ENCODING ---
        # Since we flatten the image grid into a sequence, we need to tell the
        # model where each pixel came from. This is a learnable embedding.
        # Assuming max image size compressed to roughly 20x20 grid = 400 patches
        self.pos_encoder = nn.Parameter(torch.randn(320, 1, d_model))

        # --- 3. SIMPLIFIED DECODER (Replaces GPT-2) ---
        # PyTorch has a built-in Transformer Decoder. We don't need to manually
        # write the attention blocks or layer norms.
        # H=128 -> H_feat=8
        # W=640 -> W_feat=40
        # Total patches = 8 * 40 = 320
        self.embedding = nn.Embedding(vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # The final prediction layer (LaTeX Prediction)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, images, tgt_text):
        """
        images: (Batch, 3, Height, Width)
        tgt_text: (Batch, Seq_Len) -> The LaTeX tokens
        """

        # --- ENCODER STEP ---
        # 1. Extract features with CNN
        # Input: (Batch, 3, H, W) -> Output: (Batch, 512, H/32, W/32)
        features = self.backbone(images)

        # 2. Adapt dimensions
        features = self.feature_adapter(features)  # (Batch, d_model, H', W')

        # 3. Flatten for the Transformer (The "Patch Partition" equivalent)
        # Permute to put Sequence dimension first (required by default PyTorch transformer)
        # Shape becomes: (Sequence_Length, Batch, d_model)
        B, C, H, W = features.shape
        memory = features.flatten(2).permute(2, 0, 1)  # (Seq_Len, Batch, d_model)

        # 4. Add positional encoding (broadcasting across batch)
        # We slice pos_encoder to match the actual sequence length of the image features
        seq_len = memory.size(0)
        pos_enc = self.pos_encoder[:seq_len, :, :].squeeze(1)  # (Seq_Len, d_model)
        memory = memory + pos_enc.unsqueeze(1)  # Broadcast to (Seq_Len, Batch, d_model)

        # --- DECODER STEP ---
        # 1. Embed the text (target LaTeX)
        # Shape: (Seq_Len, Batch, d_model)
        tgt_emb = self.embedding(tgt_text).permute(1, 0, 2)

        # 2. Create the "Causal Mask"
        # (Ensures the model can't see the future tokens, just like GPT-2)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(0)).to(images.device)

        # 3. Run the decoder
        # memory is the "Keys/Values" from the image
        # tgt_emb is the "Queries" from the text
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)

        # 4. Predict probabilities
        prediction = self.fc_out(output)  # (Seq_Len, Batch, Vocab_Size)

        # Permute back to (Batch, Seq_Len, Vocab_Size) for Loss calculation
        return prediction.permute(1, 0, 2)

    def generate_square_subsequent_mask(self, sz):
        # Creates a triangular mask to hide future tokens
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    model = Im2LatexModel()
    x = torch.rand(1, 3, 224, 224)
    tgt_text = torch.randint(0, 100, (1, 10))
    print(f"Output shape of model: {model(x, tgt_text).shape}")
