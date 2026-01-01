import torch
from torch import nn


class AffineTransformLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()

        self.gamma_emb = nn.Linear(32, hidden)  # 32: c1 // 4
        self.beta_emb = nn.Linear(32, hidden)

        # Initialize gamma bias to 1.0 (TF parity)
        nn.init.ones_(self.gamma_emb.bias)
        # beta bias is already initialized to 0 by default

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        gammas = self.gamma_emb(sigma).view(sigma.size(0), 1, -1)
        betas = self.beta_emb(sigma).view(sigma.size(0), 1, -1)
        return x * gammas + betas
