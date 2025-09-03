import torch
from torch import nn

class Gemma3Attention(nn.Module):
    """TODO
    """
    def __init__(self, residual_channel_size: int = 2043, query_dim: int = 2048, head_dimension: int = 256):
        super().__init__()
        self.q = nn.Linear(residual_channel_size, query_dim)