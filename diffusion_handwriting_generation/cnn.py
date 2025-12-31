import torch

from diffusion_handwriting_generation.conditioning import AffineTransformLayer


class ConvBlock(torch.nn.Module):
    """Convolutional block with skip connection and affine conditioning."""

    def __init__(
        self,
        d_inp: int,
        d_out: int,
        dils: tuple[int, int] = (1, 1),
        drop_rate: float = 0.0,
    ):
        """
        Args:
            d_inp: Number of input channels
            d_out: Number of output channels
            dils: Dilation rates for Conv1D layers
            drop_rate: Dropout rate
        """
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
        # Convert to channel-first once for convolutions to reduce transpose overhead
        x_ch = x.transpose(1, 2)  # [B, C, T]

        x_skip_ch = self.conv_skip(x_ch)

        x1_ch = self.conv1(self.act(x_ch))
        x1 = x1_ch.transpose(1, 2)  # back to [B, T, C] for affine
        x1 = self.drop(self.affine1(x1, alpha))

        x2_ch = self.conv2(self.act(x1.transpose(1, 2)))
        x2 = x2_ch.transpose(1, 2)
        x2 = self.drop(self.affine2(x2, alpha))

        x3 = self.fc(self.act(x2))
        x3 = self.drop(self.affine3(x3, alpha))

        # Bring skip connection back to channel-last and add
        x_out = x3 + x_skip_ch.transpose(1, 2)
        return x_out
