import torch
from torch import nn


class AffineTransformLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()

        self.gamma_emb = nn.Linear(32, hidden)  # 32: c1 // 4
        self.beta_emb = nn.Linear(32, hidden, bias=False)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        gammas = self.gamma_emb(sigma)
        betas = self.beta_emb(sigma)
        x = x * gammas + betas
        return x.permute(0, 2, 1)
