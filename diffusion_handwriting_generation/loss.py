import torch
import torch.nn.functional as F


def loss_fn(
    eps: torch.Tensor,
    score_pred: torch.Tensor,
    pen_lifts: torch.Tensor,
    pen_lifts_pred: torch.Tensor,
    alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss function for diffusion handwriting generation.

    Args:
        eps: Ground truth noise
        score_pred: Predicted noise (score)
        pen_lifts: Ground truth pen lift states
        pen_lifts_pred: Predicted pen lift probabilities
        alphas: Diffusion timestep alphas

    Returns:
        Tuple of (total_loss, score_loss, pen_lifts_loss)
    """
    score_loss = F.mse_loss(score_pred, eps, reduction="mean")

    pen_lifts = torch.clamp(pen_lifts, min=1e-7, max=1 - 1e-7)
    pen_lifts_loss = (
        F.binary_cross_entropy(pen_lifts_pred, pen_lifts, reduction="none").mean(dim=1)
        * alphas.squeeze(-1)
    ).mean()

    return score_loss + pen_lifts_loss, score_loss, pen_lifts_loss
