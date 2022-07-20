import torch
from torch.nn import Module

from ltron.dataset.paths import get_dataset_info

from ltron_torch.models.sequence_fcn import (
    named_resnet_independent_sequence_fcn)
from ltron_torch.models.heads import (
    LinearMultiheadDecoder, Conv2dMultiheadDecoder)

class BreakAndMakeResnet(Module):
    def __init__(self, config):
        super(BreakAndMakeResnet, self).__init__()
        modes = 23
        dataset_info = get_dataset_info(config.dataset)
        num_shapes = max(dataset_info['shape_ids'].values())+1
        num_colors = max(dataset_info['color_ids'].values())+1
        
        self.fcn = named_resnet_independent_sequence_fcn(
            'resnet50',
            256,
            global_heads = LinearMultiheadDecoder(
                2048, {'mode':modes, 'shape':num_shapes, 'color':num_colors}),
            dense_heads = Conv2dMultiheadDecoder(256, 2, kernel_size=1)
        )
    
    def forward(self, x_work, x_hand, x_r):
        global_x, workspace_x = self.fcn(x_work)
        global_x['workspace'] = workspace_x
        return global_x
