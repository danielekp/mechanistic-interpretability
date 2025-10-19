import math

import torch
import torch.nn as nn

class Gemma3TextScaledWordEmbedding(nn.Module):
    """Embedding for Gemma3 model.

    Notes:
    - The hidden size is 2304 because it is divisible for a lot of numbers (more
    than powers of two), so it is easy to explore different values for the number
    of different heads.
    - The scaled is applied to stabilize the values.
    """
    def __init__(self, vocab_size: int = 262208, hidden_size: int = 1152, padding_idx: int = 0): 
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.embedding(input_ids) * math.sqrt(self.hidden_size)

  
