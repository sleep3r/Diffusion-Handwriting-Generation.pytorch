from typing import Tuple

import torch
import torch.nn as nn


def get_angles(
    pos: torch.Tensor, i: torch.Tensor, C: int, pos_factor: float = 1
) -> torch.Tensor:
    """
    Calculates the angles for the given position and index.

    Args:
        pos (torch.Tensor): Input position with shape (batch_size, sequence_length);
        i (torch.Tensor): Index of the position;
        C (int): Maximum number of possible positions;
        pos_factor (float): Scaling factor for the position. Default is 1.

    Returns:
        torch.Tensor: Angles with shape (batch_size, sequence_length).
    """
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / float(C))
    return pos * angle_rates * pos_factor


def positional_encoding(position: int, C: int, pos_factor: float = 1) -> torch.Tensor:
    """
    Calculates the positional encoding for the given position.

    Args:
        position (int): input position;
        C (int): Maximum number of possible positions;
        pos_factor (float, optional): Scaling factor for the position. Default is 1.

    Returns:
        torch.Tensor: Positional encoding with shape (batch_size, sequence_length, C).
    """
    angle_rads = get_angles(
        torch.arange(position).unsqueeze(1),
        torch.arange(C).unsqueeze(0),
        C,
        pos_factor=pos_factor,
    )

    angle_rads[:, :, 0::2] = torch.sin(angle_rads[:, :, 0::2])
    angle_rads[:, :, 1::2] = torch.cos(angle_rads[:, :, 1::2])
    pos_encoding = angle_rads[None, ...]
    return pos_encoding.float()


def scaled_dp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates dot-product attention between `q` and `k` with scaling, then apply it to `v`.

    Args:
        q (torch.Tensor): Tensor of shape `(batch_size, d_model, seq_len_q)` representing the query;
        k (torch.Tensor): Tensor of shape `(batch_size, d_model, seq_len_k)` representing the keys;
        v (torch.Tensor): Tensor of shape `(batch_size, d_model, seq_len_k)` representing the values;
        mask (Tensor, optional): Tensor of shape `(batch_size, seq_len_q, seq_len_k)` representing the attention mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The weighted sum of `v` with attention weights and the attention weights.
        The first element has shape `(batch_size, d_model, seq_len_q)` and the second element has shape `(batch_size, seq_len_q, seq_len_k)`.
    """
    # (batch_size, d_model, seq_len_q, seq_len_k)
    qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.size(-1)
    scaled_qk = qk / torch.sqrt(dk)
    if mask is not None:
        scaled_qk += mask * -1e12

    # (batch_size, seq_len_q, seq_len_k)
    attention_weights = torch.softmax(scaled_qk, dim=-1)
    # (batch_size, d_model, seq_len_q)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, C: int, num_heads: int):
        super().__init__()

        self.C = C
        self.num_heads = num_heads
        self.wq = nn.Linear(C, C)
        self.wk = nn.Linear(C, C)
        self.wv = nn.Linear(C, C)
        self.dense = nn.Linear(C, C)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Splits heads.

        Args:
            x (torch.Tensor): tensor of shape (batch_size, seq_len, C);
            batch_size (int): batch size.

        Returns:
            torch.Tensor: tensor of shape (batch_size, num_heads, seq_len, C // num_heads)
        """
        x = x.view(batch_size, -1, self.num_heads, self.C // self.num_heads)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = self.wq(q), self.wk(k), self.wv(v)  # (bs, sl, C)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)  # (bs, nh, sl, C // nh) for q,k,v

        attention, attention_weights = scaled_dp_attn(q, k, v, mask)
        attention = attention.permute(0, 2, 1, 3)  # (bs, sl, nh, C // nh)
        concat_attention = attention.reshape(batch_size, -1, self.C)  # (bs, sl, c)
        output = self.dense(concat_attention)
        return output, attention_weights
