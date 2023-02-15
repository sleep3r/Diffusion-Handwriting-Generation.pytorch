import torch.nn as nn

from diffusion_handwriting_generation.text_style import (
    ConvSubLayer,
    DecoderLayer,
    TextStyleEncoder,
)
from diffusion_handwriting_generation.utils.nn import create_padding_mask, ff_network


class DiffusionWriter(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        c1: int = 128,
        c2: int = 192,
        c3: int = 256,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.input_dense = nn.Linear(2, c1)
        self.sigma_ffn = ff_network(1, c1 // 4, hidden=2048)
        self.enc1 = ConvSubLayer(c1, c1, [1, 2])
        self.enc2 = ConvSubLayer(c1, c2, [1, 2])
        self.enc3 = DecoderLayer(c2 * 2, c2, 3, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c2, c3, [1, 2])
        self.enc5 = DecoderLayer(c2 * 2, c3, 4, drop_rate, pos_factor=2)
        self.pool = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.skip_conv1 = nn.Conv1d(c1, c2, kernel_size=3, padding="same")
        self.skip_conv2 = nn.Conv1d(c2, c3, kernel_size=3, padding="same")
        self.skip_conv3 = nn.Conv1d(c3, c2 * 2, kernel_size=3, padding="same")
        self.text_style_encoder = TextStyleEncoder(c2 * 2, c2 * 4)
        self.att_dense = nn.Linear(c1 * 2, c2 * 2)
        self.att_layers = [
            DecoderLayer(c2 * 2, c2 * 2, num_heads=6, drop_rate=drop_rate)
            for _ in range(num_layers)
        ]

        self.dec3 = ConvSubLayer(c2 * 2, c3, [1, 2])
        self.dec2 = ConvSubLayer(c3, c2, [1, 1])
        self.dec1 = ConvSubLayer(c2, c1, [1, 1])
        self.output_dense = nn.Linear(128, 2)
        self.pen_lifts_dense = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())

    def forward(self, strokes, text, sigma, style_vector):
        """
        Args:
            strokes: [B, T, 2]
            text: [B, L]
            sigma: [B, 1, 1]
            style_vector: [B, 8, 1280]
        """
        sigma = self.sigma_ffn(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        x = self.input_dense(strokes)
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1.transpose(2, 1)).transpose(2, 1)

        h2 = self.enc2(h2, sigma)
        h2, _ = self.enc3(h2, text, sigma, text_mask)
        h3 = self.pool(h2.transpose(2, 1)).transpose(2, 1)

        h3 = self.enc4(h3, sigma)
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        x = self.pool(h3.transpose(2, 1)).transpose(2, 1)

        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)

        x = self.upsample(x.transpose(2, 1)).transpose(2, 1) + self.skip_conv3(
            h3.transpose(2, 1)
        ).transpose(2, 1)
        x = self.dec3(x, sigma)

        x = self.upsample(x.transpose(2, 1)).transpose(2, 1) + self.skip_conv2(
            h2.transpose(2, 1)
        ).transpose(2, 1)
        x = self.dec2(x, sigma)

        x = self.upsample(x.transpose(2, 1)).transpose(2, 1) + self.skip_conv1(
            h1.transpose(2, 1)
        ).transpose(2, 1)
        x = self.dec1(x, sigma)

        output = self.output_dense(x)
        pl = self.pen_lifts_dense(x)
        return output, pl, att
