"""
notation clarification:
we use the variable "alpha" for alpha_bar (cumprod 1-beta)
the alpha in the paper is replaced with 1-beta
"""

from typing import List

import torch
from torch import nn


def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set


def explin(min_val: float, max_val: float, L: int) -> torch.Tensor:
    """
    Calculates the exponential values in a logarithmic space.

    Args:
        min_val (float): minimum value of the exponential;
        max_val (float): maximum value of the exponential;
        L (int): number of values to calculate.

    Returns:
        torch.Tensor: tensor of exponential values.
    """
    log_min = torch.log(torch.tensor(min_val))
    log_max = torch.log(torch.tensor(max_val))
    lin_space = torch.linspace(log_min.item(), log_max.item(), L)
    return torch.exp(lin_space)


def get_alphas(batch_size: int, alpha_set: List[float]) -> torch.Tensor:
    """
    Returns random alpha values from the set of predefined alpha values.

    Args:
        batch_size (int): number of alpha values to generate;
        alpha_set (List[float]): set of predefined alpha values.

    Returns:
        torch.Tensor: tensor of generated alpha values.
    """
    alpha_indices = torch.randint(
        low=0, high=len(alpha_set) - 1, size=(batch_size, 1), dtype=torch.int32
    )
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    alphas = (
        torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas) + lower_alphas
    )
    alphas = alphas.unsqueeze(1).unsqueeze(2)
    return alphas


def standard_diffusion_step(
    xt: torch.Tensor,
    eps: torch.Tensor,
    beta: float,
    alpha: float,
    add_sigma: bool = True,
) -> torch.Tensor:
    """
    Performs the standard diffusion step.

    Args:
        xt (torch.Tensor): input tensor;
        eps (torch.Tensor): tensor of epsilon values;
        beta (float): beta value;
        alpha (float): alpha value;
        add_sigma (bool, optional): whether to add random noise.

    Returns:
        torch.Tensor: tensor after the standard diffusion step.
    """
    beta = torch.tensor(beta)
    alpha = torch.tensor(alpha)

    x_t_minus1 = (1 / torch.sqrt(1 - beta)) * (
        xt - (beta * eps / torch.sqrt(1 - alpha))
    )
    if add_sigma:
        x_t_minus1 += torch.sqrt(beta) * (torch.randn(xt.shape))
    return x_t_minus1


def new_diffusion_step(
    xt: torch.Tensor,
    eps: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_next: torch.Tensor,
) -> torch.Tensor:
    """
    Performs a single step of the new diffusion process, as described in the paper.

    Parameters:
    xt (torch.Tensor): input tensor;
    eps (torch.Tensor): tensor representing epsilon;
    beta (torch.Tensor): tensor representing beta;
    alpha (torch.Tensor): tensor representing alpha;
    alpha_next (torch.Tensor): tensor representing alpha_next.

    Returns:
    torch.Tensor: result of the diffusion step.
    """
    x_t_minus1 = (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    x_t_minus1 += torch.randn(xt.shape) * torch.sqrt(1 - alpha_next)
    return x_t_minus1


def reshape_up(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Reshapes the input tensor to have twice as many elements in its second dimension.

    Args:
        x (torch.Tensor): input tensor to be reshaped;
        factor (int, optional): factor to scale the second dimension by. Default is 2.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    x_shape = x.shape
    x = x.view(x_shape[0], x_shape[1] * factor, x_shape[2] // factor)
    return x


def reshape_down(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """
    Reshapes the input tensor to have half as many elements in its second dimension.

    Args:
        x (torch.Tensor): input tensor to be reshaped;
        factor (int, optional): factor to scale the second dimension by. Default is 2.

    Returns:
        torch.Tensor: reshaped tensor.
    """
    x_shape = x.shape
    x = x.reshape([x_shape[0], x_shape[1] // factor, x_shape[2] * factor])
    return x


def ff_network(
    inp: int, out: int, hidden: int = 768, act_before: bool = True
) -> nn.Sequential:
    """Builds a feedforward network in PyTorch.

    Args:
        inp (int): number of input units in the first layer of the network;
        out (int): number of output units in the final layer of the network;
        hidden (int): number of units in the hidden layer. Defaults to 768;
        act_before (bool, optional): whether to apply the activation function before the final layer. Defaults to True.

    Returns:
        nn.Sequential: feedforward network.
    """
    ff_layers = [nn.Linear(inp, hidden), nn.Linear(hidden, out), nn.ReLU()]
    if act_before:
        ff_layers.insert(0, nn.ReLU())
    return nn.Sequential(*ff_layers)


def create_padding_mask(seq: torch.Tensor, repeats: int = 1) -> torch.Tensor:
    """
    Creates a padding mask for the given sequence tensor.

    Args:
        seq (torch.Tensor): input sequence tensor with shape (batch_size, sequence_length);
        repeats (int): number of repetitions of the sequence tensor. Default is 1.

    Returns:
        torch.Tensor: padding mask with shape (batch_size, 1, 1, sequence_length * repeats).
    """
    seq = torch.eq(seq, 0).float()
    seq = seq.repeat(1, repeats, 1)
    mask = seq[:, None, :]
    return mask
