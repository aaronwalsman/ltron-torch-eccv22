import math

import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Embedding
from torch.optim import Adam, AdamW

from ltron.config import Config

from ltron_torch.models.parameter import NoWeightDecayParameter

class OptimizerConfig(Config):
    optimizer = 'adamw'
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    
    grad_norm_clip = 1.
    
    linear_warmup_cosine_decay = True
    cosine_decay_start = 5000 # TODO: I hope this is good?
    cosine_decay_stop = 500000 # TODO: I hope this is good too?
    min_learning_rate_scale = 0.1

def build_optimizer(config, model, checkpoint=None):
    
    print('learning rate: %f'%config.learning_rate)
    print('weight decay: %f'%config.weight_decay)
    print('betas: %f, %f'%config.betas)
    print('cosine decay start: %f'%config.cosine_decay_start)
    print('cosine decay stop: %f'%config.cosine_decay_stop)
    print('min learning rate scale: %f'%config.min_learning_rate_scale)
    
    decay_params = []
    no_decay_params = []
    no_decay_modules = (
        BatchNorm1d, BatchNorm2d, BatchNorm3d, LayerNorm, Embedding)
    for module_name, module in model.named_modules():
        is_no_decay_module = isinstance(module, no_decay_modules)
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = (
                '%s.%s'%(module_name, param_name)
                if module_name else param_name
            )

            if isinstance(param, NoWeightDecayParameter):
                no_decay_params.append(param)
            elif param_name.endswith('bias') or is_no_decay_module:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    optimizer_groups = [
        {'params': decay_params,
         'weight_decay':config.weight_decay,
        },
        {'params': no_decay_params,
         'weight_decay':0.,
        },
    ]
    
    if config.optimizer == 'adam':
        optimizer = Adam(
            optimizer_groups,
            lr=config.learning_rate,
            betas=config.betas,
        )
    elif config.optimizer == 'adamw':
        optimizer = AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=config.betas,
        )
    else:
        raise ValueError('Unexpected optimizer "%s"'%config.optimizer)
    
    if checkpoint is not None:
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        optimizer.load_state_dict(checkpoint)
    
    return optimizer

def clip_grad(config, model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

class NoScheduler:
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer
    
    def step(self):
        pass
    
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass

class LinearWarmupCosineDecayScheduler:
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.steps = 0
        self.base_lr = [
            param_group['lr'] for param_group in optimizer.param_groups]
    
    def step(self):
        lb = self.config.min_learning_rate_scale
        if self.steps < self.config.cosine_decay_start:
            lr_scale = self.steps / self.config.cosine_decay_start
        elif self.steps < self.config.cosine_decay_stop:
            i = self.steps - self.config.cosine_decay_start
            j = self.config.cosine_decay_stop - self.config.cosine_decay_start
            t = i/j
            lr_scale = lb + 0.5 * (1. - lb) * (1 + math.cos(t * math.pi))
        else:
            lr_scale = lb
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = lr * lr_scale
        
        self.steps += 1
    
    def state_dict(self):
        return {'steps':self.steps}
    
    def load_state_dict(self, state_dict):
        if 'steps' in state_dict:
            self.steps = state_dict['steps']
        else:
            self.steps = 0
            print('NO STEPS FOUND, MAKE THIS AN ERROR AGAIN')

def build_scheduler(config, optimizer, checkpoint=None):
    if config.linear_warmup_cosine_decay:
        scheduler = LinearWarmupCosineDecayScheduler(config, optimizer)
    else:
        scheduler = NoScheduler(config, optimizer)
    
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint)
    
    return scheduler
