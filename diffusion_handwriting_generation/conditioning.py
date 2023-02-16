import torch
from torch import nn


class AffineTransformLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()

        self.gamma_emb = nn.Linear(32, hidden)  # TODO: 32?
        self.beta_emb = nn.Linear(32, hidden)

        self.init_weights()

    def init_weights(self):
        self.gamma_emb.bias.data.fill_(1)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        gammas = self.gamma_emb(sigma)
        betas = self.beta_emb(sigma)
        return x * gammas + betas
