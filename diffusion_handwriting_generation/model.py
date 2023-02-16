import torch
import torch.nn as nn

from diffusion_handwriting_generation.attention import MultiHeadAttention, PosEmbeddings
from diffusion_handwriting_generation.cnn import ConvBlock
from diffusion_handwriting_generation.conditioning import AffineTransformLayer
from diffusion_handwriting_generation.text_style import TextStyleModel
from diffusion_handwriting_generation.utils.nn import create_padding_mask, ff_network


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_inp: int,
        d_out: int,
        num_heads: int,
        drop_rate: float = 0.1,
        pos_factor: float = 1.0,
    ):
        super().__init__()

        # Activation function
        self.act = nn.SiLU()

        # Positional embeddings
        self.text_pe = PosEmbeddings(d_out, pos_factor=pos_factor)(
            torch.arange(2000)
        )
        self.stroke_pe = PosEmbeddings(d_out, pos_factor=pos_factor)(
            torch.arange(2000)
        )

        # Dropout layer
        self.drop = nn.Dropout(drop_rate)

        # Layer normalization
        self.lnorm = nn.LayerNorm(d_out, eps=1e-6)

        # Fully-connected layers
        self.text_dense = nn.Linear(d_inp, d_out)
        self.ffn = ff_network(d_out, d_out, hidden=d_out * 2)

        # Multi-head attention
        self.mha = MultiHeadAttention(d_out, num_heads)
        self.mha2 = MultiHeadAttention(d_out, num_heads)

        # Affine transformation layers
        self.affine0 = AffineTransformLayer(d_out)
        self.affine1 = AffineTransformLayer(d_out)
        self.affine2 = AffineTransformLayer(d_out)
        self.affine3 = AffineTransformLayer(d_out)

    def forward(self, x, text, sigma, text_mask):
        text = self.text_dense(self.act(text))
        text = self.affine0(self.lnorm(text), sigma)
        text_pe = text + self.text_pe[:, : text.size(1)]

        x_pe = x + self.stroke_pe[:, : x.size(1)]
        x2, att = self.mha(x_pe, text_pe, text, text_mask)
        x2 = self.lnorm(self.drop(x2))
        x2 = self.affine1(x2, sigma) + x

        x2_pe = x2 + self.stroke_pe[:, : x2.size(1)]
        x3, _ = self.mha2(x2_pe, x2_pe, x2)
        x3 = self.lnorm(x2 + self.drop(x3))
        x3 = self.affine2(x3, sigma)

        x4 = self.ffn(x3)
        x4 = self.drop(x4) + x3
        out = self.affine3(self.lnorm(x4), sigma)
        return out, att


class DiffusionModel(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        c1: int = 128,
        c2: int = 192,
        c3: int = 256,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        # Input layer
        self.input_dense = nn.Linear(2, c1)

        # Sigma feedforward network
        self.sigma_ffn = ff_network(1, c1 // 4, hidden=2048, act_before=False)

        # Encoder layers
        self.enc1 = ConvBlock(c1, c1, dils=[1, 2])
        self.enc2 = ConvBlock(c1, c2, dils=[1, 2])
        self.enc3 = DecoderLayer(
            c2 * 2, c2, num_heads=3, drop_rate=drop_rate, pos_factor=4
        )
        self.enc4 = ConvBlock(c2, c3, dils=[1, 2])
        self.enc5 = DecoderLayer(
            c2 * 2, c3, num_heads=4, drop_rate=drop_rate, pos_factor=2
        )

        # Pooling and upsampling
        self.pool = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Skip connections and text style encoder
        self.skip_conv1 = nn.Conv1d(c1, c2, kernel_size=3, padding="same")
        self.skip_conv2 = nn.Conv1d(c2, c3, kernel_size=3, padding="same")
        self.skip_conv3 = nn.Conv1d(c3, c2 * 2, kernel_size=3, padding="same")
        self.text_style_model = TextStyleModel(c2 * 2, c2 * 4)

        # Attention layers
        self.att_dense = nn.Linear(c1 * 2, c2 * 2)
        self.att_layers = nn.ModuleList(
            [
                DecoderLayer(c2 * 2, c2 * 2, num_heads=6, drop_rate=drop_rate)
                for _ in range(num_layers)
            ]
        )

        # Decoder layers
        self.dec3 = ConvBlock(c2 * 2, c3, dils=[1, 2])
        self.dec2 = ConvBlock(c3, c2, dils=[1, 1])
        self.dec1 = ConvBlock(c2, c1, dils=[1, 1])

        # Output layer
        self.output_dense = nn.Linear(c1, 2)

        # Pen lifts layer
        self.pen_lifts_dense = nn.Sequential(nn.Linear(c1, 2), nn.Sigmoid())

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
        text = self.text_style_model(text, style_vector, sigma)

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
