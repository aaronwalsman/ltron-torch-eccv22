import torch

from ltron.gym.envs.ltron_env import async_ltron
from ltron.gym.envs.blocks_env import BlocksEnvConfig, BlocksEnv

from ltron_torch.dataset.episode_dataset import (
    EpisodeDatasetConfig, build_episode_loader,
)
from ltron_torch.models.hand_table_transformer import (
    HandTableTransformerConfig,
    HandTableTransformer,
)
from ltron_torch.models.hand_table_lstm import (
    HandTableLSTMConfig,
    HandTableLSTM,
)
from ltron_torch.interface.blocks import BlocksInterfaceConfig
from ltron_torch.interface.blocks_hand_table_transformer import (
    BlocksHandTableTransformerInterface,
)
from ltron_torch.interface.blocks_hand_table_lstm import (
    BlocksHandTableLSTMInterface,
)
from ltron_torch.train.optimizer import OptimizerConfig, build_optimizer
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, behavior_cloning,
)

class BlocksBCConfig(
    EpisodeDatasetConfig,
    BlocksEnvConfig,
    BlocksInterfaceConfig,
    HandTableTransformerConfig,
    HandTableLSTMConfig,
    OptimizerConfig,
    BehaviorCloningConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    dataset = 'blocks'
    
    num_test_envs = 4
    
    num_modes = 6
    num_shapes = 6
    num_colors = 6
    
    cursor_channels = 1

def train_blocks_bc(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BlocksBCConfig.from_commandline()
    
    print('-'*80)
    print('Building Model')
    if config.model == 'transformer':
        model = HandTableTransformer(config).to(torch.device(config.device))
        interface = BlocksHandTableTransformerInterface(model, config)
    elif config.model == 'lstm':
        model = HandTableLSTM(config).to(torch.device(config.device))
        interface = BlocksHandTableLSTMInterface(model, config)
    else:
        raise ValueError(
            'config "model" parameter must be either "transformer" or "lstm"')
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(model, config)
    
    print('-'*80)
    print('Building Data Loader')
    train_loader = build_episode_loader(config)
    
    print('-'*80)
    print('Building Test Env')
    test_env = async_ltron(config.num_test_envs, BlocksEnv, config)
    
    behavior_cloning(
        config, model, optimizer, train_loader, test_env, interface)

def sanity_check_transformer():
    
    # some imports that I don't want to polute the main module
    from ltron.hierarchy import index_hierarchy
    import numpy
    
    # load the config
    config = BlocksTransformerBCConfig.from_commandline()
    config.shuffle=False
    config.batch_size=1
    device = torch.device(config.device)
    
    # build the model and interface
    model = BreakAndMakeTransformer(config).to(device)
    model.eval()
    interface = BlocksTransformerInterface(model, config)
    
    # build the loader
    train_loader = build_sequence_train_loader(config)
    
    # peel off the first batch
    batch, pad = next(iter(train_loader))
    
    # do a bunch of tests
    with torch.no_grad():
        
        # make sure multiple passes with a single frame produce the same results
        pad0 = numpy.ones(config.batch_size, numpy.long)
        tensors0 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0]), pad0)
        
        x_table0a, x_hand0a, x_token0a = model(*tensors0)
        x_table0b, x_hand0b, x_token0b = model(*tensors0)
        assert torch.allclose(x_table0a, x_table0b)
        assert torch.allclose(x_hand0a, x_hand0b)
        assert torch.allclose(x_token0a, x_token0b)
        
        # make sure a pass with a single frame matches the first frame of a
        # pass with two frames
        pad01 = numpy.ones(config.batch_size, numpy.long)*2
        tensors01 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0,1]), pad01)
        x_table01, x_hand01, x_token01 = model(*tensors01)
        assert torch.allclose(x_table0a, x_table01[[0]], atol=1e-6)
        assert torch.allclose(x_hand0a, x_hand01[[0]], atol=1e-6)
        assert torch.allclose(x_token0a, x_token01[[0]], atol=1e-6)
        
        # make sure a pass with two frames matches the first two frames of a
        # pass with three frames
        pad012 = numpy.ones(config.batch_size, numpy.long)*3
        tensors012 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0,1,2]), pad012)
        x_table012, x_hand012, x_token012 = model(*tensors012)
        assert torch.allclose(x_table01, x_table012[[0,1]], atol=1e-6)
        assert torch.allclose(x_hand01, x_hand012[[0,1]], atol=1e-6)
        assert torch.allclose(x_token01, x_token012[[0,1]], atol=1e-6)
        
        # make sure the results computed from an entire sequence match the
        # results computed frame-by-frame with use-memory
        tensors = interface.observation_to_tensors(batch['observations'], pad)
        x_table_seq, x_hand_seq, x_token_seq = model(*tensors)
        
        x_tables = []
        x_hands = []
        x_tokens = []
        seq_len = numpy.max(pad)
        total_tiles = 0
        total_tokens = 0
        for i in range(seq_len):
            print('-'*80)
            print(i)
            seq_obs = index_hierarchy(batch['observations'], [i])
            
            i_pad = numpy.ones(pad.shape, dtype=numpy.long)
            i_tensors = interface.observation_to_tensors(seq_obs, i_pad)
            if i == 0:
                use_memory = torch.zeros(pad.shape, dtype=torch.long).to(device)
            else:
                use_memory = torch.ones(pad.shape, dtype=torch.long).to(device)
            
            total_tiles += i_tensors[0].shape[0]
            total_tokens += i_tensors[4].shape[0]
            
            xi_table, xi_hand, xi_token = model(
                *i_tensors, use_memory=use_memory)
            x_tables.append(xi_table)
            x_hands.append(xi_hand)
            x_tokens.append(xi_token)
            
        x_table_cat = torch.cat(x_tables, dim=0)
        x_hand_cat = torch.cat(x_hands, dim=0)
        x_token_cat = torch.cat(x_tokens, dim=0)
        
    try:
        assert torch.allclose(x_table_seq, x_table_cat, atol=1e-6)
        assert torch.allclose(x_hand_seq, x_hand_cat, atol=1e-6)
        assert torch.allclose(x_token_seq, x_token_cat, atol=1e-6)
    except AssertionError:
        print('BORK')
        import pdb
        pdb.set_trace()
        
    print('we did it!')
