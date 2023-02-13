import torch
import torch.nn as nn
import torchvision.models as models

from diffusion_handwriting_generation.attention import (
    MultiHeadAttention,
    positional_encoding,
)


def reshape_up(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Reshapes the input tensor to have twice as many elements in its second dimension.

    Args:
        x (torch.Tensor): input tensor to be reshaped;
        factor (int, optional): factor to scale the second dimension by. Default is 2.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    x_shape = x.shape
    x = x.view(x_shape[0], x_shape[1] * factor, x_shape[2] // factor)
    return x


def reshape_down(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Reshapes the input tensor to have half as many elements in its second dimension.

    Args:
        x (torch.Tensor): input tensor to be reshaped;
        factor (int, optional): factor to scale the second dimension by. Default is 2.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    x_shape = x.shape
    x = x.reshape([x_shape[0], x_shape[1] // factor, x_shape[2] * factor])
    return x


def ff_network(C: int, dff: int = 768, act_before: bool = True) -> nn.Sequential:
    """Builds a feedforward network in PyTorch.

    Args:
        C (int): The number of output units in the final layer of the network;
        dff (int, optional): The number of units in the hidden layer. Defaults to 768;
        act_before (bool, optional): Whether to apply the activation function before the final layer. Defaults to True.

    Returns:
        nn.Sequential: feedforward network.
    """
    ff_layers = [nn.Linear(dff, C), nn.ReLU()]
    if act_before:
        ff_layers.insert(0, nn.ReLU())
    return nn.Sequential(*ff_layers)


class AffineTransformLayer(nn.Module):
    """Applies an affine transformation to the input tensor."""

    def __init__(self, filters: int):
        super().__init__()
        self.gamma_emb = nn.Linear(in_features=filters, bias=True)
        self.beta_emb = nn.Linear(in_features=filters, bias=True)

        self.gamma_emb.bias.data.fill_(1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape (batch_size, filters, seq_len);
            sigma (torch.Tensor): scaling parameters with shape (batch_size, seq_len).

        Returns:
            torch.Tensor: transformed tensor with shape (batch_size, filters, seq_len)
        """
        gammas = self.gamma_emb(sigma)
        betas = self.beta_emb(sigma)
        return x * gammas + betas


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        drop_rate: float = 0.1,
        pos_factor: float = 1.0,
    ):
        super().__init__()

        self.text_pe = positional_encoding(2000, d_model, pos_factor=1)
        self.stroke_pe = positional_encoding(2000, d_model, pos_factor=pos_factor)
        self.drop = nn.Dropout(drop_rate)
        self.lnorm = nn.LayerNorm(d_model, eps=1e-6)
        self.text_dense = nn.Linear(d_model, d_model)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = ff_network(d_model, d_model * 2)
        self.affine0 = AffineTransformLayer(d_model)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)

    def forward(self, x, text, sigma, text_mask):
        text = self.text_dense(nn.SiLU()(text))
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


def get_activation(activation: str = "relu") -> nn.Module:
    """
    Returns an activation function.

    Args:
        activation (str, optional): activation function. Default is "relu".

    Returns:
        nn.Module: activation function.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return nn.SiLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unknown activation function.")


class ConvSubLayer(nn.Module):
    def __init__(
        self,
        filters: int,
        dils: list = (1, 1),
        activation: str = "swish",
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.act = get_activation(activation)
        self.affine1 = AffineTransformLayer(filters // 2)
        self.affine2 = AffineTransformLayer(filters)
        self.affine3 = AffineTransformLayer(filters)
        self.conv_skip = nn.Conv1d(filters, 3, padding=1)
        self.conv1 = nn.Conv1d(filters // 2, 3, dilation=dils[0], padding=1)
        self.conv2 = nn.Conv1d(filters, 3, dilation=dils[1], padding=1)
        self.fc = nn.Linear(filters, filters)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x, alpha):
        """
        Args:
            filters (int): number of filters;
            dils (list): dilation rates for Conv1D layers;
            activation (str): activation function to use;
            drop_rate (float): dropout rate.

        Returns:
        - output: Tensor, shape (batch_size, seq_len, filters)
        """
        x_skip = self.conv_skip(x)
        x = self.conv1(self.act(x))
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.act(x))
        x = self.drop(self.affine2(x, alpha))
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))
        x += x_skip
        return x


class StyleExtractor(nn.Module):
    """
    Takes a grayscale image (with the last channel) with pixels [0, 255].
    Rescales to [-1, 1] and repeats along the channel axis for 3 channels.
    Uses a MobileNetV2 with pretrained weights from imagenet as initial weights.
    """

    def __init__(self):
        super(StyleExtractor, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True, progress=True)
        self.mobilenet.features[0][0].in_channels = 1
        self.local_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.freeze_all_layers()

    def freeze_all_layers(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): tensor of shape (batch_size, 1, 224, 224).

        Returns:
            torch.Tensor: tensor of shape (batch_size, 1280).
        """
        x = torch.tensor(img, dtype=torch.float32)
        x = (x / 127.5) - 1
        x = torch.cat((x, x, x), dim=1)
        x = self.mobilenet.features(x)
        x = self.local_pool(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class TextStyleEncoder(nn.Module):
    def __init__(self, d_model, d_ff=512):
        super().__init__()

        self.emb = nn.Embedding(73, d_model)
        self.text_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.style_ffn = ff_network(d_model, d_ff)
        self.mha = MultiHeadAttention(d_model, 8)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(0.3)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_ffn = ff_network(d_model, d_model * 2)

    def forward(self, text, style, sigma):
        style = reshape_up(self.dropout(style), 5)
        style = self.affine1(self.layernorm(self.style_ffn(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layernorm(text), sigma)
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layernorm(text + mha_out), sigma)
        text_out = self.affine4(self.layernorm(self.text_ffn(text)), sigma)
        return text_out
