import torch
import torch.nn as nn

class Gemma3RotaryEmbedding(nn.Module):
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
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        self.theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self._cos_cached = None
        self._sin_cached = None

    def _build_cache(self, device: torch.device, dtype: torch.dtype):

        t = torch.arange(self.max_seq_len, device=device, dtype=self.theta.dtype)
        freqs = torch.outer(t, self.theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        self._cos_cached = freqs_cis.real.to(dtype)
        self._sin_cached = freqs_cis.imag.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._cos_cached is None or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._build_cache(device=x.device, dtype=x.dtype)

        # Input and output shape: [batch_size, sequence_length, num_query_heads, head_dim]
        seq_len = x.shape[1]
        cos = self._cos_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = self._sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)

        # The new shape [1, T, 1, D_h/2] will broadcast across the batch and head dimensions.
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        x1, x2 = x.chunk(2, dim=-1)

        return torch.cat(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
        )