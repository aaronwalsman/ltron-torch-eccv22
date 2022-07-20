import torch

from ltron.hierarchy import map_hierarchies

from ltron_torch.models.simple_fcn import named_resnet_fcn

class IndependentSequenceFCN(torch.nn.Module):
    def __init__(self, fcn):
        super(IndependentSequenceFCN, self).__init__()
        self.fcn = fcn
    
    def forward(self, x):
        s, b, c, h, w = x.shape
        x = x.view(s*b, c, h, w)
        x = self.fcn(x)
        def reshape_to_sequence(xx):
            sb, *c = xx.shape
            return xx.view(s, b, *c)
        return map_hierarchies(reshape_to_sequence, x)

def named_resnet_independent_sequence_fcn(
    name,
    decoder_channels,
    global_heads=None,
    dense_heads=None,
    pretrained=False,
):
    fcn = named_resnet_fcn(
        name,
        decoder_channels,
        global_heads=global_heads,
        dense_heads=dense_heads,
        pretrained=pretrained,
    )
    return IndependentSequenceFCN(fcn)
