import math

import torch
import torch.nn.functional as F


class PosEmbeddings(torch.nn.Module):
    def __init__(self, dim: int, pos_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.pos_factor = pos_factor

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # time: [batch_size] or [seq_len] (usually arange)
        # We want to use the device of the input tensor
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        # [half_dim]
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # time[:, None] -> [B, 1] or [L, 1]
        # embeddings[None, :] -> [1, half_dim]
        # result: [B/L, half_dim]
        embeddings = time[:, None] * embeddings[None, :] * self.pos_factor

        # [B/L, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # Match expected output format [1, B/L, dim] or [B, L, dim] ?
        # The legacy code did: embeddings = embeddings[None, ...] -> [1, L, dim]
        # In model.py it is used as: text_pe[:, : text.size(1)]
        # So we probably want [1, L, dim] so it broadcasts over batch
        return embeddings[None, ...]


def scaled_dp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention.

    Args:
        q: [batch_size, num_heads, seq_len_q, depth]
        k: [batch_size, num_heads, seq_len_k, depth]
        v: [batch_size, num_heads, seq_len_k, depth]
        mask: Optional mask. If provided, should be broadcastable to [batch_size, num_heads, seq_len_q, seq_len_k].
              Legacy code used additive mask (-1e12 for masked values).
    """
    # Use PyTorch 2.0+ optimized attention if available and no special mask requirements
    # Note: F.scaled_dot_product_attention expects:
    #   query: (N, ..., L, E)
    #   key:   (N, ..., S, E)
    #   value: (N, ..., S, E)
    #   attn_mask: (N, ..., L, S) - boolean or float additive mask

    # Check if we can use optimized implementation
    # Legacy mask was additive (-1e12). F.scaled_dot_product_attention handles that if is_causal=False

    # We will stick to manual implementation to ensure exact behavior reproduction
    # and compatibility with the specific mask format from legacy code,
    # but cleaned up.

    # (batch_size, num_heads, seq_len_q, seq_len_k)
    qk = torch.matmul(q, k.transpose(-2, -1))
    dk = k.size(-1)
    scaled_qk = qk / math.sqrt(dk)

    if mask is not None:
        scaled_qk += mask * -1e12

    # (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights = torch.softmax(scaled_qk, dim=-1)

    # (batch_size, num_heads, seq_len_q, depth)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, C: int, num_heads: int):
        super().__init__()
        self.C = C
        self.num_heads = num_heads

        self.wq = torch.nn.Linear(C, C)
        self.wk = torch.nn.Linear(C, C)
        self.wv = torch.nn.Linear(C, C)
        self.dense = torch.nn.Linear(C, C)

    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, C]
        Returns:
            [batch_size, num_heads, seq_len, C // num_heads]
        """
        x = x.view(batch_size, -1, self.num_heads, self.C // self.num_heads)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        # Linear projections
        q = self.wq(q)  # (bs, sl, C)
        k = self.wk(k)  # (bs, sl, C)
        v = self.wv(v)  # (bs, sl, C)

        # Split heads -> (bs, nh, sl, C // nh)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Attention
        attention, attention_weights = scaled_dp_attn(q, k, v, mask)

        # Concatenate heads
        # (bs, nh, sl, C // nh) -> (bs, sl, nh, C // nh)
        attention = attention.permute(0, 2, 1, 3)
        # -> (bs, sl, C)
        concat_attention = attention.reshape(batch_size, -1, self.C)

        # Output projection
        output = self.dense(concat_attention)

        return output, attention_weights
