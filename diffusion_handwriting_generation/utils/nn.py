"""
notation clarification:
we use the variable "alpha" for alpha_bar (cumprod 1-beta)
the alpha in the paper is replaced with 1-beta
"""

import math

import torch


def get_device() -> str:
    """Returns the appropriate device string (cuda or cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_beta_set() -> torch.Tensor:
    """Returns the predefined beta schedule for diffusion."""
    return 0.02 + explin(1e-5, 0.4, 60)


def explin(min_val: float, max_val: float, l: int) -> torch.Tensor:
    """
    Calculates exponentially-spaced values in logarithmic space.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        l: Number of values

    Returns:
        Tensor of exponential values
    """
    log_min = math.log(min_val)
    log_max = math.log(max_val)
    lin_space = torch.linspace(log_min, log_max, l)
    return torch.exp(lin_space)


def get_alphas(batch_size: int, alpha_set: torch.Tensor) -> torch.Tensor:
    """
    Samples random alpha values from the alpha schedule.

    Args:
        batch_size: Number of alpha values to generate
        alpha_set: Set of predefined alpha values

    Returns:
        Tensor of sampled alpha values
    """
    alpha_indices = torch.randint(
        low=0,
        high=len(alpha_set) - 1,
        size=(batch_size, 1),
        dtype=torch.int64,
    )
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    return torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas) + lower_alphas


def standard_diffusion_step(
    xt: torch.Tensor,
    eps: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    add_sigma: bool = True,
) -> torch.Tensor:
    """
    Performs standard DDPM diffusion sampling step.

    Args:
        xt: Current noisy sample
        eps: Predicted noise
        beta: Beta value for current timestep
        alpha: Alpha_bar value for current timestep
        add_sigma: Whether to add stochastic noise

    Returns:
        Denoised sample at t-1
    """
    x_t_minus1 = (1 / torch.sqrt(1 - beta)) * (xt - (beta * eps / torch.sqrt(1 - alpha)))
    if add_sigma:
        x_t_minus1 += torch.sqrt(beta) * torch.randn_like(xt)
    return x_t_minus1


def new_diffusion_step(
    xt: torch.Tensor,
    eps: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_next: torch.Tensor,
) -> torch.Tensor:
    """
    Performs alternative diffusion sampling step.

    Args:
        xt: Current noisy sample
        eps: Predicted noise
        beta: Beta value for current timestep
        alpha: Alpha_bar value for current timestep
        alpha_next: Alpha_bar value for next timestep

    Returns:
        Denoised sample at t-1
    """
    x_t_minus1 = (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    x_t_minus1 += torch.randn_like(xt) * torch.sqrt(1 - alpha_next)
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
    return x.reshape(x_shape[0], x_shape[1] * factor, x_shape[2] // factor)


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
    return x.reshape([x_shape[0], x_shape[1] // factor, x_shape[2] * factor])


def ff_network(
    inp: int,
    out: int,
    hidden: int = 768,
    act_before: bool = True,
) -> torch.nn.Sequential:
    """
    Builds a feedforward network.

    Uses SiLU/Swish activation to match TensorFlow implementation.

    Args:
        inp: Number of input features
        out: Number of output features
        hidden: Number of hidden units
        act_before: Whether to apply activation before first layer

    Returns:
        Sequential feedforward network
    """
    layers = []
    if act_before:
        layers.append(torch.nn.SiLU())
    layers.extend(
        [
            torch.nn.Linear(inp, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, out),
        ]
    )
    return torch.nn.Sequential(*layers)


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
    seq = seq.unsqueeze(1).unsqueeze(2)
    return seq.repeat(1, 1, 1, repeats)
