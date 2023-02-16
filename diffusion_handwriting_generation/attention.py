import math
from typing import Tuple

import torch
import torch.nn as nn


class PosEmbeddings(nn.Module):
    def __init__(self, dim, pos_factor=1.0):
        super().__init__()

        self.dim = dim
        self.pos_factor = pos_factor

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * self.pos_factor
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings[None, ...]
        return embeddings


def scaled_dp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates dot-product attention between `q` and `k` with scaling, then apply it to `v`.

    Args:
        q (torch.Tensor): tensor of shape `(batch_size, d_model, seq_len_q)` representing the query;
        k (torch.Tensor): tensor of shape `(batch_size, d_model, seq_len_k)` representing the keys;
        v (torch.Tensor): tensor of shape `(batch_size, d_model, seq_len_k)` representing the values;
        mask (Tensor, optional): tensor of shape `(batch_size, seq_len_q, seq_len_k)` representing the attention mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: weighted sum of `v` with attention weights and the attention weights.
        The first element has shape `(batch_size, d_model, seq_len_q)` and the second element has shape `(batch_size, seq_len_q, seq_len_k)`.
    """
    # (batch_size, d_model, seq_len_q, seq_len_k)
    qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.size(-1)
    scaled_qk = qk / math.sqrt(dk)
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
