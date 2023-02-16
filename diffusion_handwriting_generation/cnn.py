import torch
from torch import nn

from diffusion_handwriting_generation.conditioning import AffineTransformLayer
from diffusion_handwriting_generation.utils.nn import get_activation


class ConvBlock(nn.Module):
    """
    Args:
        filters (int): number of filters;
        dils (list): dilation rates for Conv1D layers;
        activation (str): activation function to use;
        drop_rate (float): dropout rate.
    """

    def __init__(
        self,
        d_inp: int,
        d_out: int,
        dils: list = (1, 1),
        activation: str = "swish",
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # Activation function
        self.act = get_activation(activation)

        # Affine transformation layers
        self.affine1 = AffineTransformLayer(d_out // 2)
        self.affine2 = AffineTransformLayer(d_out)
        self.affine3 = AffineTransformLayer(d_out)

        # Convolutional layers
        self.conv_skip = nn.Conv1d(
            d_inp,
            d_out,
            kernel_size=3,
            padding="same",
        )
        self.conv1 = nn.Conv1d(
            d_inp,
            d_out // 2,
            kernel_size=3,
            dilation=dils[0],
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            d_out // 2,
            d_out,
            kernel_size=3,
            dilation=dils[1],
            padding="same",
        )

        # Fully-connected layer
        self.fc = nn.Linear(d_out, d_out)

        # Dropout layer
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, alpha):
        x_skip = self.conv_skip(x.transpose(2, 1)).transpose(2, 1)

        x = self.conv1(self.act(x.transpose(2, 1))).transpose(2, 1)
        x = self.drop(self.affine1(x, alpha))
        x = self.conv2(self.act(x.transpose(2, 1))).transpose(2, 1)
        x = self.drop(self.affine2(x, alpha))
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))

        x += x_skip
        return x
