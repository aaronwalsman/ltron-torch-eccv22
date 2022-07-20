import torch
from torch.nn import Module

from ltron_torch.models.parameter import NoWeightDecayParameter

class LearnedPositionalEncoding(Module):
    def __init__(self, channels, max_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = NoWeightDecayParameter(
            torch.zeros(max_length, channels))
    
    def forward(self, i):
        s, b = i.shape
        c = self.encoding.shape[-1]
        return self.encoding[i.reshape(-1)].view(s,b,c)
