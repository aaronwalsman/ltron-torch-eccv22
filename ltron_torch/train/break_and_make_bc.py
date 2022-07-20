import random
import argparse

import numpy

import torch

from conspiracy.log import SynchronousConsecutiveLog

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig, BreakAndMakeEnv)

from ltron_torch.dataset.break_and_make_dataset import (
    BreakAndMakeDatasetConfig, BreakAndMakeDataset, BreakOnlyDataset,
)
from ltron_torch.dataset.episode_dataset import build_episode_loader
from ltron_torch.models.hand_table_transformer import (
    HandTableTransformerConfig,
    HandTableTransformer,
)
from ltron_torch.models.stubnet_transformer import (
    StubnetTransformerConfig,
    StubnetTransformer,
)
from ltron_torch.models.hand_table_lstm import (
    HandTableLSTMConfig,
    HandTableLSTM,
)
from ltron_torch.interface.break_and_make import BreakAndMakeInterface
from ltron_torch.interface.break_and_make_hand_table_transformer import (
    BreakAndMakeHandTableTransformerInterfaceConfig,
    BreakAndMakeHandTableTransformerInterface,
)
from ltron_torch.interface.break_and_make_stubnet_transformer import (
    BreakAndMakeStubnetTransformerInterfaceConfig,
    BreakAndMakeStubnetTransformerInterface,
)
from ltron_torch.interface.break_and_make_hand_table_lstm import (
    BreakAndMakeHandTableLSTMInterface,
)
from ltron_torch.train.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, behavior_cloning,
)

# TODO: This file is very similar to blocks_bc.py but with different
# defaults and a different interface.  These could probably be consolidated,
# but I'm sick of tearing up everything every five minutes.  Also, let's wait
# and see what the different interface configs look like and how easy it would
# be to reconcile them.  I also need a better way for owner configs to
# conditionally overwrite stuff they inherit.  For example table_channels below
# is 2 for BreakAndMake but 1 for Blocks, and I don't want / can't have that
# specified in the config file.  It should be overwritten based on the task,
# using set_dependents, but this is error-prone because if earlier
# set_dependents calls use the old value, then things might get really messy.
# One option is just to call set_dependents again after the override?  It would
# be nice to have a simple mechanism for this kind of thing though.  Maybe a new
# overrides method or something?  Whatever, avoid the issue for now.

class BreakAndMakeBCConfig(
    BreakAndMakeDatasetConfig,
    BreakAndMakeEnvConfig,
    BreakAndMakeHandTableTransformerInterfaceConfig,
    BreakAndMakeStubnetTransformerInterfaceConfig,
    HandTableTransformerConfig,
    StubnetTransformerConfig,
    HandTableLSTMConfig,
    OptimizerConfig,
    BehaviorCloningConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    load_checkpoint = None
    use_checkpoint_config = False
    
    dataset = 'random_construction_6_6'
    train_split = 'train_episodes'
    test_split = 'test'
    
    task = 'break_and_make'
    
    num_test_envs = 4
    
    #num_modes = 23 # 7 + 7 + 3 + 2 + 1 + 2 + 1 (+2 for factored cursor)
    factor_cursor_distribution = False
    num_shapes = 6
    num_colors = 6
    
    table_channels = 2
    hand_channels = 2
    
    async_ltron = True
    
    seed = 1234567890
    
    allow_snap_flip = False

def train_break_and_make_bc(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeBCConfig.from_commandline()

    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if config.load_checkpoint is not None:
        print('-'*80)
        print('Loading Checkpoint')
        checkpoint = torch.load(config.load_checkpoint)
        # this is bad because it overwrites things specified on the command line
        # if you want to do this, find a better way, and put it in the Config
        # class itself (which is hard because the Config class is in ltron and 
        # doesn't know about pytorch
        # ok here's the compromise: I just added "use_checkpoint_config" which
        # turns on this behavior
        if config.use_checkpoint_config:
            assert 'config' in checkpoint, (
                '"config" not found in checkpoint: %s'%config.load_checkpoint)
            config = BreakAndMakeBCConfig(**checkpoint['config'])
        model_checkpoint = checkpoint['model']
        if config.train_frequency:
            optimizer_checkpoint = checkpoint['optimizer']
        else:
            optimizer_checkpoint = None
        scheduler_checkpoint = checkpoint['scheduler']
        train_log_checkpoint = checkpoint.get('train_log', None)
        test_log_checkpoint = checkpoint.get('test_log', None)
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None
        train_log_checkpoint = None
        test_log_checkpoint = None
        start_epoch = 1
    
    if config.factor_cursor_distribution:
        config.num_modes = 25
    else:
        config.num_modes = 23
    
    if config.allow_snap_flip:
        config.num_modes += 4
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Model (%s)'%config.model)
    if config.model == 'transformer':
        model = HandTableTransformer(config, model_checkpoint).to(device)
    elif config.model == 'stubnet':
        model = StubnetTransformer(config, model_checkpoint).to(device)
    elif config.model == 'lstm':
        model = HandTableLSTM(config, model_checkpoint).to(device)
    else:
        raise ValueError(
            'config "model" parameter ("%s") must be either '
            '"transformer", "stubnet" or "lstm"'%config.model
        )
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(config, model, optimizer_checkpoint)
    
    print('-'*80)
    print('Building Interface (%s)'%config.model)
    if config.model == 'transformer':
        interface = BreakAndMakeHandTableTransformerInterface(
            config, model, optimizer)
    elif config.model == 'stubnet':
        interface = BreakAndMakeStubnetTransformerInterface(
            config, model, optimizer)
    elif config.model == 'lstm':
        interface = BreakAndMakeHandTableLSTMInterface(
            config, model, optimizer)
    
    print('-'*80)
    print('Building Logs')
    train_log = interface.make_train_log(train_log_checkpoint)
    test_log = interface.make_test_log(test_log_checkpoint)
    
    print('-'*80)
    print('Building Scheduler')
    scheduler = build_scheduler(config, optimizer, scheduler_checkpoint)
    
    print('-'*80)
    print('Building Data Loader')
    train_config = BreakAndMakeBCConfig.translate(config, split='train_split')
    if config.task == 'break_and_make':
        train_dataset = BreakAndMakeDataset(train_config)
    elif config.task == 'break_only':
        train_dataset = BreakOnlyDataset(train_config)
    else:
        assert False, 'bad task: "%s"'%config.task
    train_loader = build_episode_loader(train_config, train_dataset)
    
    print('-'*80)
    print('Building Test Env')
    test_config = BreakAndMakeBCConfig.translate(config, split='test_split')
    if config.async_ltron:
        vector_ltron = async_ltron
    else:
        vector_ltron = sync_ltron
    test_env = vector_ltron(
        config.num_test_envs,
        BreakAndMakeEnv,
        test_config,
        print_traceback=True,
    )

    behavior_cloning(
        config,
        model,
        optimizer,
        scheduler,
        train_loader,
        test_env,
        interface,
        train_log,
        test_log,
        start_epoch=start_epoch,
    )

def plot_break_and_make_bc(checkpoint=None):
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str)
        args = parser.parse_args()
        checkpoint = args.checkpoint
    
    data = torch.load(open(checkpoint, 'rb'), map_location='cpu')
    train_log = BreakAndMakeInterface.make_train_log()
    train_log.set_state(data['train_log'])
    test_log = BreakAndMakeInterface.make_test_log()
    test_log.set_state(data['test_log'])
    
    train_chart = train_log.plot_grid(
        topline=True, legend=True, minmax_y=True, height=40, width=72)
    print('='*80)
    print('Train Plots')
    print(train_chart)
    
    test_chart = test_log.plot_sequential(
        legend=True, minmax_y=True, height=60, width=160)
    print('='*80)
    print('Test Plots')
    print('-'*80)
    print(test_chart)

def eval_break_and_make_bc(checkpoint=None):
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.add_argument
