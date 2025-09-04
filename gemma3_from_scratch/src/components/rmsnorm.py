import torch
import torch.nn as nn

class Gemma3RMSNorm(nn.Module):
    """Normalization layer for Gemma3 model.

    Notes:
    - The creators hypothesized that re-centering is not essential and can be skipped.
    """
    def __init__(self, dim: int, eps: float = 1e-06, dtype=torch.float32):
        super().__init__()

        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float32 = x.to(torch.float32)
        rms = torch.rsqrt(x_float32.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normalized = x_float32 * rms
        return (self.weight * x_normalized).to(input_dtype)

    