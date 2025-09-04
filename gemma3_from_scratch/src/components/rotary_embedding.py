import torch
from torch import nn

class Gemma3RotaryEmbedding(nn.module):
    """Rotary embedding for Gemma3 model.

    Notes:
    - RoPE rotates the query and key vectors based on their absolute position.
    It rotates the vectors by a certain degree based on their position.
    - It is applied to the query and and key vectors, rather than at the beggining,
    this lets the model split the embedding with the positional information.
    It is called after the input has been projected and reshaped to separate heads.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Precompute sin and cos up to max_seq_len
        t = torch.arange(max_seq_len, device=self.theta.device)
        freqs = torch.outer(t, self.theta) # Shape: [max_seq_len, dim / 2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # Shape: [max_seq_len, dim / 2]
        self.register_buffer("cos_cached", freqs_cis.real)
        self.register_buffer("sin_cached", freqs_cis.imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input and output shape: [batch_size, sequence_length, num_query_heads, head_dim]
        """
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        # The new shape [1, T, 1, D_h/2] will broadcast across the batch and head dimensions.
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        x1, x2 = x.chunk(2, dim=-1)
        rotated_x = torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )
        return rotated_x