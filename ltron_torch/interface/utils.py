import torch
from torch.distributions import Bernoulli, Categorical

def bernoulli_or_max(logits, mode):
    if mode == 'sample':
        distribution = Bernoulli(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return (logits > 0.).long()
    else:
        raise NotImplementedError

def categorical_or_max(logits, mode):
    if mode == 'sample':
        distribution = Categorical(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError

def categorical_or_max_2d(logits, mode):
    *dims, h, w = logits.shape
    logits = logits.view(*dims, h*w)
    yx = categorical_or_max(logits, mode)
    y = torch.div(yx, w, rounding_mode='floor')
    x = yx % w
    return y, x
