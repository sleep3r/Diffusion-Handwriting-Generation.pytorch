import torch


def loss_fn(
    eps: torch.Tensor,
    score_pred: torch.Tensor,
    pl: torch.Tensor,
    pl_pred: torch.Tensor,
    abar: torch.Tensor,
    bce: callable,
) -> torch.Tensor:
    score_loss = torch.mean(torch.sum(torch.pow(eps - score_pred, 2), dim=-1))
    pl_loss = torch.mean(bce(pl, pl_pred) * torch.squeeze(abar, -1))
    return score_loss + pl_loss
