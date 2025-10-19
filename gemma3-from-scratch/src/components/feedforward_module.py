import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchGELUTanh(nn.Module):
    """Smooth implementation of FELU activation function.

    Notes:
    - The formula is: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is the precise formula for the GELU approximation.
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))
    

class Gemma3MLP(nn.Module):
    """The Gated Feed-Forward Network (MLP) for the Gemma3 model.
    """
    def __init__(self, residual_channel_size: int = 1152, intermediate_size: int = 6912):
        super().__init__()

        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(residual_channel_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(residual_channel_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(self.intermediate_size, residual_channel_size, bias=False)

        self.act_fn = PytorchGELUTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        gated_output = self.act_fn(gate_output) * up_output
        
        output = self.down_proj(gated_output)
        
        return output