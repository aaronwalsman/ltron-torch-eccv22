#!/usr/bin/env python
import torch

'''
from ltron_torch.models.standard_models import single_step_model

model = single_step_model(
    'simple_fcn',
    202,
    pose_head=True,
    pretrained=True,
).cuda()

x = torch.zeros(50, 3, 256, 256).cuda()

y_hat = model(x)
'''

from ltron_torch.models.standard_models import seq_model

model = seq_model(
    'simple_fcn',
    196,
    pose_head=True,
    pretrained=True,
).cuda()

x = torch.zeros(64, 1, 3, 256, 256).cuda()
y_hat = model(x)

input()
