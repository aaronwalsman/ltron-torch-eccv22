from ltron.dataset.paths import get_dataset_info

from ltron.config import Config

from ltron_torch.train.optimizer import build_optimizer
from ltron_torch.dataset.break_and_make import (
    build_test_env,
    build_seq_train_loader,
)
from ltron_torch.models.break_and_make_lstm import (
    build_model as build_lstm_model,
)
from ltron_torch.envs.break_and_make_lstm import BreakAndMakeLSTMInterface
from ltron_torch.train.behavior_cloning import behavior_cloning

# config definitions ===========================================================

class BreakAndMakeLSTMBCConfig(Config):
    epochs=10
    batch_size=4
    num_envs=16
    loader_workers=8
    test_rollout_steps_per_epoch=256
    max_episode_length=64
    
    task = 'break_and_make'
    
    optimizer='adamw'
    learning_rate=3e-4
    weight_decay=0.1
    betas=(0.9, 0.95)
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    tile_width=16
    tile_height=16
    
    randomize_viewpoint=True,
    randomize_colors=True,
    
    dataset='omr_clean'
    train_split='rollouts'
    train_subset=None
    test_split='pico'
    test_subset=None
    
    resnet_name='resnet50'
    
    test_frequency=1
    checkpoint_frequency=10
    visualization_frequency=1
    visualization_seqs=10
    
    def set_dependents(self):
        dataset_info = get_dataset_info(self.dataset)
        self.num_classes = max(dataset_info['shape_ids'].values()) + 1
        self.num_colors = max(dataset_info['color_ids'].values()) + 1
        
        self.test_batch_rollout_steps_per_epoch = (
            self.test_rollout_steps_per_epoch // self.num_envs
        )

# train functions ==============================================================

def train_break_and_make_lstm_bc(config):
    print('='*80)
    print('Break and Make Setup')
    model = build_lstm_model(config)
    optimizer = build_optimizer(model, config)
    train_loader = build_seq_train_loader(config)
    test_env = build_test_env(config)
    interface = BreakAndMakeLSTMInterface(config)
    
    # run behavior cloning
    behavior_cloning(
        config, model, optimizer, train_loader, test_env, interface)
