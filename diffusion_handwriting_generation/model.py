import torch

from diffusion_handwriting_generation.attention import MultiHeadAttention, PosEmbeddings
from diffusion_handwriting_generation.cnn import ConvBlock
from diffusion_handwriting_generation.conditioning import AffineTransformLayer
from diffusion_handwriting_generation.text_style import TextStyleEncoder
from diffusion_handwriting_generation.utils.nn import create_padding_mask, ff_network


class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_inp: int,
        d_out: int,
        num_heads: int,
        drop_rate: float = 0.1,
        pos_factor: float = 1.0,
    ):
        super().__init__()

        self.act = torch.nn.SiLU()
        self.text_pe_gen = PosEmbeddings(d_out, pos_factor=1.0)
        self.stroke_pe_gen = PosEmbeddings(d_out, pos_factor=pos_factor)
        self.drop = torch.nn.Dropout(drop_rate)
        self.lnorm = torch.nn.LayerNorm(d_out, eps=1e-6, elementwise_affine=False)
        self.text_dense = torch.nn.Linear(d_inp, d_out)
        self.ffn = ff_network(d_out, d_out, hidden=d_out * 2)
        self.mha = MultiHeadAttention(d_out, num_heads)
        self.mha2 = MultiHeadAttention(d_out, num_heads)
        self.affine0 = AffineTransformLayer(d_out)
        self.affine1 = AffineTransformLayer(d_out)
        self.affine2 = AffineTransformLayer(d_out)
        self.affine3 = AffineTransformLayer(d_out)

    def forward(self, x, text, sigma, text_mask):
        """Expects [B, T, C] for both x and text."""
        text = self.text_dense(self.act(text))
        text = self.affine0(self.lnorm(text), sigma)

        text_pe = self.text_pe_gen(torch.arange(text.size(1), device=text.device))
        text_pe = text + text_pe

        x_pe = self.stroke_pe_gen(torch.arange(x.size(1), device=x.device))
        x_pe = x + x_pe

        x2, att = self.mha(x_pe, text_pe, text, text_mask)
        x2 = self.lnorm(self.drop(x2))
        x2 = self.affine1(x2, sigma) + x

        x2_pe = x2 + self.stroke_pe_gen(torch.arange(x2.size(1), device=x2.device))
        x3, _ = self.mha2(x2_pe, x2_pe, x2)
        x3 = self.lnorm(x2 + self.drop(x3))
        x3 = self.affine2(x3, sigma)

        x4 = self.ffn(x3)
        x4 = self.drop(x4) + x3
        out = self.affine3(self.lnorm(x4), sigma)
        return out, att


class DiffusionModel(torch.nn.Module):
    """Diffusion model for handwriting generation conditioned on text and style."""

    def __init__(
        self,
        num_layers: int = 4,
        c1: int = 128,
        c2: int = 192,
        c3: int = 256,
        drop_rate: float = 0.1,
    ):
        """
        Args:
            num_layers: Number of attention layers
            c1: Base channel dimension
            c2: Intermediate channel dimension
            c3: Bottleneck channel dimension
            drop_rate: Dropout probability
        """
        super().__init__()

        self.input_dense = torch.nn.Linear(2, c1)
        self.sigma_ffn = ff_network(1, c1 // 4, hidden=2048)

        # Encoder/decoder conv blocks work in [B, C, T]
        self.enc1 = ConvBlock(c1, c1, dils=(1, 2))
        self.enc2 = ConvBlock(c1, c2, dils=(1, 2))
        self.enc3 = EncoderLayer(c2 * 2, c2, num_heads=3, drop_rate=drop_rate, pos_factor=4)
        self.enc4 = ConvBlock(c2, c3, dils=(1, 2))
        self.enc5 = EncoderLayer(c2 * 2, c3, num_heads=4, drop_rate=drop_rate, pos_factor=2)

        # Pool/upsample work in [B, C, T]
        self.pool = torch.nn.AvgPool1d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")

        # Skip convs work in [B, C, T]
        self.skip_conv1 = torch.nn.Conv1d(c1, c2, kernel_size=3, padding="same")
        self.skip_conv2 = torch.nn.Conv1d(c2, c3, kernel_size=3, padding="same")
        self.skip_conv3 = torch.nn.Conv1d(c3, c2 * 2, kernel_size=3, padding="same")

        self.text_style_model = TextStyleEncoder(c2 * 2, c2 * 4)

        self.att_dense = torch.nn.Linear(c1 * 2, c2 * 2)
        self.att_layers = torch.nn.ModuleList(
            [
                EncoderLayer(c2 * 2, c2 * 2, num_heads=6, drop_rate=drop_rate)
                for _ in range(num_layers)
            ],
        )

        self.dec3 = ConvBlock(c2 * 2, c3, dils=(1, 2))
        self.dec2 = ConvBlock(c3, c2, dils=(1, 1))
        self.dec1 = ConvBlock(c2, c1, dils=(1, 1))

        self.output_dense = torch.nn.Linear(c1, 2)
        self.pen_lifts_dense = torch.nn.Sequential(
            torch.nn.Linear(c1, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, strokes, text, sigma, style_vector):
        """
        Forward pass of the diffusion model.

        Args:
            strokes: Noisy stroke coordinates [B, T, 2]
            text: Text token indices [B, L]
            sigma: Diffusion timestep embeddings [B, 1]
            style_vector: Style features from reference image [B, 14, 1280]

        Returns:
            Tuple of (predicted_noise, pen_lift_probs, None)
        """
        sigma = self.sigma_ffn(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_model(text, style_vector, sigma)

        # Input: [B, T, 2] -> [B, T, C1]
        x = self.input_dense(strokes)
        # Convert to [B, C, T] for conv blocks
        x = x.transpose(1, 2)

        # Encoder path (all in [B, C, T])
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)
        # enc3 needs [B, T, C]
        h2_t = h2.transpose(1, 2)
        h2_t, _ = self.enc3(h2_t, text, sigma, text_mask)
        h2 = h2_t.transpose(1, 2)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)
        # enc5 needs [B, T, C]
        h3_t = h3.transpose(1, 2)
        h3_t, _ = self.enc5(h3_t, text, sigma, text_mask)
        h3 = h3_t.transpose(1, 2)
        x = self.pool(h3)

        # Attention layers need [B, T, C]
        x = x.transpose(1, 2)
        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, _ = att_layer(x, text, sigma, text_mask)

        # Decoder path (back to [B, C, T])
        x = x.transpose(1, 2)
        x = self.upsample(x) + self.skip_conv3(h3)
        x = self.dec3(x, sigma)

        x = self.upsample(x) + self.skip_conv2(h2)
        x = self.dec2(x, sigma)

        x = self.upsample(x) + self.skip_conv1(h1)
        x = self.dec1(x, sigma)

        # Output: [B, C, T] -> [B, T, C] -> [B, T, 2]
        x = x.transpose(1, 2)
        output = self.output_dense(x)
        pen_lifts = self.pen_lifts_dense(x).squeeze(-1)
        return output, pen_lifts, None
