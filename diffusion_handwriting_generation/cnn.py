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

        self.act = torch.nn.SiLU()

        self.affine1 = AffineTransformLayer(d_out // 2)
        self.affine2 = AffineTransformLayer(d_out)
        self.affine3 = AffineTransformLayer(d_out)

        # Conv1d expects [B, C, T]
        self.conv_skip = torch.nn.Conv1d(d_inp, d_out, kernel_size=3, padding="same")
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
            dilation=dils[0],
            padding="same",
        )

        self.fc = torch.nn.Linear(d_out, d_out)
        self.drop = torch.nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor):
        """
        Forward pass. Expects [B, C, T] input, returns [B, C, T].

        Args:
            x: Input tensor [B, C, T]
            alpha: Conditioning alpha [B, 1, C_alpha]

        Returns:
            Output tensor [B, C, T]
        """
        # Skip connection
        x_skip = self.conv_skip(x)

        # First conv path
        x = self.conv1(self.act(x))
        # Convert to [B, T, C] for affine
        x = x.transpose(1, 2).contiguous()
        x = self.drop(self.affine1(x, alpha))

        # Back to [B, C, T] for conv
        x = x.transpose(1, 2).contiguous()
        x = self.conv2(self.act(x))

        # To [B, T, C] for affine
        x = x.transpose(1, 2).contiguous()
        x = self.drop(self.affine2(x, alpha))

        # FC and final affine
        x = self.fc(self.act(x))
        x = self.drop(self.affine3(x, alpha))

        # Back to [B, C, T] and add skip
        x = x.transpose(1, 2).contiguous()
        x += x_skip
        return x
