import math

import torch


class PosEmbeddings(torch.nn.Module):
    def __init__(self, dim, pos_factor=1.0):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        embeddings = embeddings.to(self.device)
        return embeddings


def scaled_dp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
