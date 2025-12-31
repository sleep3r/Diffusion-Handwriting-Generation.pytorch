import torch

from diffusion_handwriting_generation.conditioning import AffineTransformLayer


class ConvBlock(torch.nn.Module):
    """
    Args:
        d_inp (int): number of input channels;
        d_out (int): number of output channels;
        dils (Tuple[int, int]): dilation rates for Conv1D layers;
        drop_rate (float): dropout rate.
    """

    def __init__(
        self,
        d_inp: int,
        d_out: int,
        dils: tuple[int, int] = (1, 1),
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # Activation function
        self.act = torch.nn.SiLU()

        # Affine transformation layers
        self.affine1 = AffineTransformLayer(d_out // 2)
        self.affine2 = AffineTransformLayer(d_out)
        self.affine3 = AffineTransformLayer(d_out)

        # Convolutional layers
        self.conv_skip = torch.nn.Conv1d(
            d_inp,
            d_out,
            kernel_size=3,
            padding="same",
        )
        self.conv1 = torch.nn.Conv1d(
            d_inp,
            d_out // 2,
            kernel_size=3,
            dilation=dils[0],
            padding="same",
        )
        self.conv2 = torch.nn.Conv1d(
            d_out // 2,
            d_out,
            kernel_size=3,
            dilation=dils[1],
            padding="same",
        )

        # Fully-connected layer
        self.fc = torch.nn.Linear(d_out, d_out)

        # Dropout layer
        self.drop = torch.nn.Dropout(drop_rate)

    def _conv(self, x: torch.Tensor, conv: torch.nn.Conv1d) -> torch.Tensor:
        return conv(x.transpose(2, 1)).transpose(2, 1)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor):
        x_skip = self._conv(x, self.conv_skip)

        x = self._conv(self.act(x), self.conv1)
        x = self.drop(self.affine1(x, alpha))

        x = self._conv(self.act(x), self.conv2)
        x = self.drop(self.affine2(x, alpha))

        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))

        x += x_skip
        return x
