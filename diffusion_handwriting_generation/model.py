import torch
import torch.nn as nn

from diffusion_handwriting_generation.text_style import (
    ConvSubLayer,
    DecoderLayer,
    TextStyleEncoder,
    ff_network,
)


def create_padding_mask(seq: torch.Tensor, repeats: int = 1) -> torch.Tensor:
    """
    Creates a padding mask for the given sequence tensor.

    Args:
        seq (torch.Tensor): Input sequence tensor with shape (batch_size, sequence_length).
        repeats (int, optional): Number of repetitions of the sequence tensor. Default is 1.

    Returns:
        torch.Tensor: Padding mask with shape (batch_size, 1, 1, sequence_length * repeats).
    """
    seq = torch.eq(seq, 0).float()
    seq = seq.repeat(1, repeats, 1)
    mask = seq[:, None, None, :]
    return mask


def loss_fn(
    eps: torch.Tensor,
    score_pred: torch.Tensor,
    pl: torch.Tensor,
    pl_pred: torch.Tensor,
    abar: torch.Tensor,
    bce: callable,
) -> torch.Tensor:
    """
    Computes the loss for the Transformer model.

    Args:
        eps (torch.Tensor): target scores;
        score_pred (torch.Tensor): predicted scores;
        pl (torch.Tensor): target label positions;
        pl_pred (torch.Tensor): predicted label positions;
        abar (torch.Tensor): mask for label positions;
        bce (callable): binary cross entropy loss function.

    Returns:
        torch.Tensor: model loss.
    """
    score_loss = torch.mean(torch.sum(torch.pow(eps - score_pred, 2), dim=-1))
    pl_loss = torch.mean(bce(pl, pl_pred) * torch.squeeze(abar, -1))
    return score_loss + pl_loss


class DiffusionWriter(nn.Module):
    def __init__(
        self, num_layers=4, c1=128, c2=192, c3=256, drop_rate=0.1, num_heads=8
    ):
        super(DiffusionWriter, self).__init__()
        self.input_dense = nn.Linear(c1)
        self.sigma_ffn = ff_network(c1 // 4, 2048)
        self.enc1 = ConvSubLayer(c1, [1, 2])
        self.enc2 = ConvSubLayer(c2, [1, 2])
        self.enc3 = DecoderLayer(c2, 3, drop_rate, pos_factor=4)
        self.enc4 = ConvSubLayer(c3, [1, 2])
        self.enc5 = DecoderLayer(c3, 4, drop_rate, pos_factor=2)
        self.pool = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.skip_conv1 = nn.Conv1d(c2, 3, padding=1)
        self.skip_conv2 = nn.Conv1d(c3, 3, padding=1)
        self.skip_conv3 = nn.Conv1d(c2 * 2, 3, padding=1)
        self.text_style_encoder = TextStyleEncoder(c2 * 2, c2 * 4)
        self.att_dense = nn.Linear(c2 * 2)
        self.att_layers = [
            DecoderLayer(c2 * 2, 6, drop_rate) for _ in range(num_layers)
        ]

        self.dec3 = ConvSubLayer(c3, [1, 2])
        self.dec2 = ConvSubLayer(c2, [1, 1])
        self.dec1 = ConvSubLayer(c1, [1, 1])
        self.output_dense = nn.Linear(2)
        self.pen_lifts_dense = nn.Sequential(nn.Linear(2), nn.Sigmoid())

    def forward(self, strokes, text, sigma, style_vector):
        sigma = self.sigma_ffn(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        x = self.input_dense(strokes)
        h1 = self.enc1(x, sigma)
        h2 = self.pool(h1)

        h2 = self.enc2(h2, sigma)
        h2, _ = self.enc3(h2, text, sigma, text_mask)
        h3 = self.pool(h2)

        h3 = self.enc4(h3, sigma)
        h3, _ = self.enc5(h3, text, sigma, text_mask)
        x = self.pool(h3)

        x = self.att_dense(x)
        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)

        x = self.upsample(x) + self.skip_conv3(h3)
        x = self.dec3(x, sigma)

        x = self.upsample(x) + self.skip_conv2(h2)
        x = self.dec2(x, sigma)

        x = self.upsample(x) + self.skip_conv1(h1)
        x = self.dec1(x, sigma)

        output = self.output_dense(x)
        pl = self.pen_lifts_dense(x)
        return output, pl, att
