import torch

from ltron_torch.dataset.break_and_make import (
    build_test_env, build_sequence_train_loader)
from ltron_torch.train.behavior_cloning import BehaviorCloningConfig
from ltron_torch.train.optimizer import OptimizerConfig, build_optimizer
from ltron_torch.train.behavior_cloning import behavior_cloning
from ltron_torch.models.break_and_make_transformer import (
    BreakAndMakeTransformerConfig,
    BreakAndMakeTransformer,
    BreakAndMakeTransformerInterface,
)

class BreakAndMakeTransformerBCConfig(
    BehaviorCloningConfig, OptimizerConfig, BreakAndMakeTransformerConfig
):
    device = 'cuda'

def train_break_and_make_transformer_bc(config):
    print('='*80)
    print('Break And Make Setup')
    model = BreakAndMakeTransformer(config).to(torch.device(config.device))
    optimizer = build_optimizer(model, config)
    train_loader = build_sequence_train_loader(config)
    test_env = build_test_env(config)
    interface = BreakAndMakeTransformerInterface(model, config)
    
    behavior_cloning(
        config, model, optimizer, train_loader, test_env, interface)
