import torch
from torch import nn


def loss_fn(
    eps: torch.Tensor,
    score_pred: torch.Tensor,
    pen_lifts: torch.Tensor,
    pen_lifts_pred: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    bce = nn.BCELoss()
    score_loss = torch.mean(torch.sum(torch.pow(eps - score_pred, 2), dim=-1))
    pen_lifts_loss = torch.mean(bce(pen_lifts, pen_lifts_pred) * torch.squeeze(alphas, -1))

    print({'score_loss': score_loss, 'pen_lifts_loss': pen_lifts_loss})
    return score_loss + pen_lifts_loss
