import torch
from torch import nn

class Gemma3RMSNorm(nn.Module):
    """Normalization layer for Gemma3 model.

    Notes:
    - The creators hypothesized that re-centering is not essential and can be skipped.
    """
    def __init__(self, dim: int, eps: float = 1e-06):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x

    