import math

import torch


class PosEmbeddings(torch.nn.Module):
    """Sinusoidal positional embeddings."""

    def __init__(self, dim: int, pos_factor: float = 1.0):
        super().__init__()

        self.dim = dim
        self.pos_factor = pos_factor

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings for given time steps."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * self.pos_factor
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings[None, ...]


def scaled_dp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None]:
    """
    Scaled dot-product attention using PyTorch's optimized SDPA.

    Args:
        q: Query tensor [batch, heads, seq_len_q, depth]
        k: Key tensor [batch, heads, seq_len_k, depth]
        v: Value tensor [batch, heads, seq_len_k, depth]
        mask: Optional additive mask (1 for positions to mask out)

    Returns:
        Tuple of (attention output, None)
    """
    attn_mask = mask * -1e9 if mask is not None else None
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return output, None


class MultiHeadAttention(torch.nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.dense = torch.nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Apply multi-head attention.

        Args:
            q: Query tensor [batch, seq_len, d_model]
            k: Key tensor [batch, seq_len, d_model]
            v: Value tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Tuple of (output, None)
        """
        batch_size = q.shape[0]

        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        attn_output, _ = scaled_dp_attn(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        output = self.dense(attn_output)

        return output, None
