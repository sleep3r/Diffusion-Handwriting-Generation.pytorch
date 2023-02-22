import torch
from torch import nn


def loss_fn(
    eps: torch.Tensor,
    score_pred: torch.Tensor,
    pen_lifts: torch.Tensor,
    pen_lifts_pred: torch.Tensor,
    alphas: torch.Tensor,
) -> torch.Tensor:
    score_loss = torch.mean(torch.sum(torch.pow(eps - score_pred, 2), dim=-1))

    pen_lifts = torch.clamp(pen_lifts, min=1e-7, max=1 - 1e-7)
    pen_lifts_loss = torch.mean(
        nn.BCELoss(reduction="none")(pen_lifts, pen_lifts_pred).mean(dim=1)
        * torch.squeeze(alphas, -1)
    )
    return score_loss + pen_lifts_loss
