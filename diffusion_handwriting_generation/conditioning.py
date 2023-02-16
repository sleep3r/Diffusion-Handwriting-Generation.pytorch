import torch
from torch import nn


class AffineTransformLayer(nn.Module):
    """Applies an affine transformation to the input tensor."""

    def __init__(self, hidden: int):
        super().__init__()

        self.gamma_emb = nn.Linear(32, hidden)  # TODO: 32?
        self.beta_emb = nn.Linear(32, hidden)

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
