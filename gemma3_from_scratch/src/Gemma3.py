import torch
from torch import nn

from .components import Gemma3TextScaledWordEmbedding

class Gemma3(nn.module):
    def __init__(self):
        self.embedding = Gemma3TextScaledWordEmbedding()