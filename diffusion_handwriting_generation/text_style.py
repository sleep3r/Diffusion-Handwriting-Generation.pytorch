import numpy as np
import torch
from torch import nn
from torchvision import models

from diffusion_handwriting_generation.attention import MultiHeadAttention
from diffusion_handwriting_generation.conditioning import AffineTransformLayer
from diffusion_handwriting_generation.utils.nn import ff_network, reshape_up


class StyleExtractor(nn.Module):
    """Extracts style features from handwriting images using MobileNetV2."""

    def __init__(self):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT,
            progress=True,
        ).to(self.device)
        self.local_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.freeze_all_layers()

    def freeze_all_layers(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, img_batch: np.ndarray) -> torch.Tensor:
        """Extract style features from grayscale images."""
        x = torch.tensor(img_batch, dtype=torch.float32).to(self.device)
        x = (x / 127.5) - 1
        x = x.repeat(1, 3, 1, 1)

        x = self.mobilenet.features(x)
        x = self.local_pool(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)


class TextStyleEncoder(nn.Module):
    """Encodes text with style conditioning using cross-attention."""

    def __init__(self, d_model, d_ff=512):
        super().__init__()

        self.d_model = d_model

        # Embedding layer
        self.emb = nn.Embedding(73, d_model)

        # Feed-forward layers
        self.style_ffn = ff_network(256, d_model, hidden=d_ff)  # 256: 1280 / 5
        self.text_ffn = ff_network(d_model, d_model, hidden=d_model * 2)

        # Attention layer and normalization
        self.mha = MultiHeadAttention(d_model, 8)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Affine layers
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)

    def forward(self, text, style, sigma):
        style = reshape_up(self.dropout(style), 5)

        style = self.style_ffn(style)
        style = self.layernorm(style)
        style = self.affine1(style, sigma)

        text = self.emb(text)
        text = self.layernorm(text)
        text = self.affine2(text, sigma)
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layernorm(text + mha_out), sigma)
        text_out = self.affine4(self.layernorm(self.text_ffn(text)), sigma)
        return text_out
