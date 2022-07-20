import json

import torch
from torch.nn import Linear, Conv2d, ModuleDict, ModuleList

from ltron.hierarchy import map_hierarchies

class MultiheadDecoder(torch.nn.Module):
    def __init__(self, heads):
        super(MultiheadDecoder, self).__init__()
        self.heads = heads
    
    def forward(self, x):
        def multi_forward(module):
            return module(x)
        return map_hierarchies(
            multi_forward,
            self.heads,
            InDictClass=ModuleDict,
            InListClass=ModuleList,
        )

def ModuleMultiheadDecoder(Module, in_channels, heads, *args, **kwargs):
    if isinstance(heads, str):
        heads = json.loads(heads)
    
    def make_linear(head_channels):
        return Module(in_channels, head_channels, *args, **kwargs)
    heads = map_hierarchies(
        make_linear, heads, OutDictClass=ModuleDict, OutListClass=ModuleList)
    return MultiheadDecoder(heads)

def LinearMultiheadDecoder(*args, **kwargs):
    return ModuleMultiheadDecoder(Linear, *args, **kwargs)

def Conv2dMultiheadDecoder(*args, **kwargs):
    return ModuleMultiheadDecoder(Conv2d, *args, **kwargs)
