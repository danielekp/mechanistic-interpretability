import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Gemma3TextScaledWordEmbedding, Gemma3RotaryEmbedding, Gemma3Attention, Gemma3MLP, Gemma3RMSNorm
from .gemma3_decoder import Gemma3DecoderLayer

class Gemma3TextModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.vocab_size = config['vocab_size']
        self.residual_channel_size = config['residual_channel_size']

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            self.vocab_size, self.residual_channel_size, padding_idx=0
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config) for _ in range(config['num_hidden_layers'])]
        )
        self.norm = Gemma3RMSNorm(self.residual_channel_size )
        
        self.rotary_emb = Gemma3RotaryEmbedding(dim=config['head_dimension'])
        self.rotary_emb_local = Gemma3RotaryEmbedding(dim=config['head_dimension'])

    def _make_causal_mask(self, input_ids_shape, device):
        batch_size, seq_len = input_ids_shape
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        attn_mask = self._make_causal_mask(input_ids.shape, hidden_states.device)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                rotary_emb=self.rotary_emb,
                attn_mask=attn_mask
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
