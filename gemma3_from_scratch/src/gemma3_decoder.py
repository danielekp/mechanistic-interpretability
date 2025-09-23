from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Gemma3RotaryEmbedding, Gemma3Attention, Gemma3MLP, Gemma3RMSNorm

class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.self_attn = Gemma3Attention(
            residual_channel_size=config['residual_channel_size'],
            query_dim=config['query_dim'],
            total_kv_dim=config['total_kv_dim'],
            head_dimension=config['head_dimension']
        )
        self.mlp = Gemma3MLP(
            residual_channel_size=config['residual_channel_size'],
            intermediate_size=config['intermediate_size']
        )
        self.input_layernorm = Gemma3RMSNorm(config['residual_channel_size'])
        self.post_attention_layernorm = Gemma3RMSNorm(config['residual_channel_size'])
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config['residual_channel_size'])
        self.post_feedforward_layernorm = Gemma3RMSNorm(config['residual_channel_size'])

    def forward(
        self, 
        hidden_states: torch.Tensor,
        rotary_emb: Gemma3RotaryEmbedding,
        attn_mask: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(normed_hidden_states, rotary_emb, attn_mask)
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output # First residual connection
        
        residual = hidden_states
        
        normed_hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        normed_mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + normed_mlp_output # Second residual connection
        
        return hidden_states