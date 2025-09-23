import torch
import torch.nn as nn

from .gemma3_textmodel import Gemma3TextModel

class Gemma3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Gemma3TextModel(config)
        self.lm_head = nn.Linear(config['residual_channel_size'], config['vocab_size'], bias=False)

        # A crucial optimization: the language model head weights are tied to the
        # embedding weights. This saves a lot of memory and improves performance.
        self.model.embed_tokens.embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        
        logits = self.lm_head(hidden_states)
        
        return logits