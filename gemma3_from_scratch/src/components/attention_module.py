import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import Gemma3RMSNorm
from .rotary_embedding import Gemma3RotaryEmbedding

class Gemma3Attention(nn.Module):
    """Attention layer for Gemma3 model.

    Notes
    - The following is a special case of Grouped-Query Attention called Multi-Query Attention (MQA).
    """
    def __init__(self, residual_channel_size: int = 1152, query_dim: int = 1024, total_kv_dim: int = 256, head_dimension: int = 256):
        super().__init__()

        self.residual_channel_size = residual_channel_size
        self.query_dim = query_dim
        self.total_kv_dim = total_kv_dim
        self.head_dimension = head_dimension

        self.num_query_heads = query_dim // head_dimension
        self.num_kv_heads = total_kv_dim // head_dimension

        self.q_proj = nn.Linear(residual_channel_size, query_dim, bias=False)
        self.k_proj = nn.Linear(residual_channel_size, total_kv_dim, bias=False)
        self.v_proj = nn.Linear(residual_channel_size, total_kv_dim, bias=False)
        
        self.output_proj = nn.Linear(query_dim, residual_channel_size, bias=False)
        
        self.q_norm = Gemma3RMSNorm(head_dimension)
        self.k_norm = Gemma3RMSNorm(head_dimension)

    def forward(self, residual_channels: torch.Tensor, positional_emb: Gemma3RotaryEmbedding, attn_mask: torch.Tensor = None):
        batch_size, seq_len, _ = residual_channels.shape

        queries = self.q_proj(residual_channels)
        keys = self.k_proj(residual_channels)
        values = self.v_proj(residual_channels)

        # Reshape Q, K, V to separate the heads
        # [B, T, total_dim] -> [B, T, num_heads (or num_kv_heads), head_dim]
        queries = queries.view(batch_size, seq_len, self.num_query_heads, self.head_dimension)
        keys = keys.view(batch_size, seq_len, self.num_kv_heads, self.head_dimension)
        values = values.view(batch_size, seq_len, self.num_kv_heads, self.head_dimension)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        queries = positional_emb(queries)
        keys = positional_emb(keys)

        if self.num_kv_heads != self.num_query_heads:
            keys = keys.repeat_interleave(self.num_query_heads // self.num_kv_heads, dim=2)
            values = values.repeat_interleave(self.num_query_heads // self.num_kv_heads, dim=2)

        # Transpose for attention calculation: [B, T, H, D_h] -> [B, H, T, D_h]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        # (B, H, T, D_h) @ (B, H, D_h, T) -> (B, H, T, T)
        attn_scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dimension)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_output = torch.matmul(attn_weights, values)

        # Reshape output back to the original format
        # (B, H, T, D_h) -> (B, T, H, D_h) -> (B, T, H * D_h)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.query_dim)

        return self.output_proj(attn_output)
