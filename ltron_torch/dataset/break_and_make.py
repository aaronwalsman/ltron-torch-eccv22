import os

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

import ltron.settings as settings
from ltron.config import Config
from ltron.exceptions import LtronException
from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import BreakAndMakeEnv
from ltron.gym.rollout_storage import RolloutStorage
from ltron.hierarchy import (
    index_hierarchy,
    stack_numpy_hierarchies,
    concatenate_numpy_hierarchies,
    #auto_pad_stack_numpy_hierarchies,
)
from ltron.plan.plannerd import Roadmap, RoadmapPlanner
from ltron.bricks.brick_scene import BrickScene

from ltron_torch.dataset.collate import pad_stack_collate

# config =======================================================================

class BreakAndMakeDatasetConfig(Config):
    num_seqs=16
    num_envs=1
    dataset='random_six'
    collection='random_six'
    train_split='simple_single'
    rollout_directory='rollouts'
    #train_subset=None
    subset=None
    split='all'
    start = 0
    end = None
    file_override=None
    
    table_image_width=256
    table_image_height=256
    table_map_width=64
    table_map_height=64
    hand_image_width=96
    hand_image_height=96
    hand_map_width=24
    hand_map_height=24
    
    randomize_viewpoint=True,
    randomize_colors=True,
    
    max_episode_length=32
    
    error_handling='record'

error_handling = 'raise'

# data generation ==============================================================

def generate_offline_dataset(dataset_config):
    
    dataset_info = get_dataset_info(dataset_config.dataset)
    shape_ids = dataset_info['shape_ids']
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(
        dataset_config.dataset,
        dataset_config.split,
        subset=dataset_config.subset,
    )
    
    #print('-'*80)
    #print('Initializing storage')
    #rollout_storage = RolloutStorage(dataset_config.num_envs)
    #observation = train_env.reset()
    #terminal = numpy.ones(dataset_config.num_envs, dtype=numpy.bool)
    #reward = numpy.zeros(dataset_config.num_envs)
    
    collection = dataset_config.dataset # ugh, wrong, but fix later
    episode_path = os.path.join(
        settings.collections[collection],
        'episodes',
    )
    
    num_paths = len(dataset_paths['mpd'])
    if dataset_config.end is None:
        end = num_paths
    else:
        end = dataset_config.end
    if dataset_config.start is None:
        start = 0
    else:
        start = dataset_config.start
    print('-'*80)
    print('Planning plans %i-%i'%(start, end))
    iterate = tqdm.tqdm(range(start, end))
    complete_seqs = 0
    problems = {}
    for i in iterate:
        '''
        env = break_and_make_env(
            dataset=dataset_config.dataset,
            split=dataset_config.train_split,
            subset=dataset_config.train_subset,
            rank=i,
            size=num_paths,
            table_image_width=dataset_config.table_image_width,
            table_image_height=dataset_config.table_image_height,
            hand_image_width=dataset_config.hand_image_width,
            hand_image_height=dataset_config.hand_image_height,
            table_map_width=dataset_config.table_map_width,
            table_map_height=dataset_config.table_map_height,
            hand_map_width=dataset_config.hand_map_width,
            hand_map_height=dataset_config.hand_map_height,
            max_episode_length=512,
            check_collisions=True,
            print_traceback=True,
            dataset_reset_mode='single_pass',
            randomize_viewpoint=dataset_config.randomize_viewpoint,
            randomize_colors=dataset_config.randomize_colors,
            include_score=False,
            train=True,
        )
        '''
        
        env = BreakAndMakeEnv(
            dataset_config, rank=i, size=num_paths, print_traceback=True)
        
        observation_seq = []
        action_seq = []
        
        # get the full assembly and state --------------------------------------
        start_observation = env.reset()
        full_assembly = start_observation['table_assembly']
        full_state = env.get_state()
        
        # build the break goal (empty) assembly --------------------------------
        max_instances = full_assembly['shape'].shape[0]
        max_edges = full_assembly['edges'].shape[1]
        empty_assembly = {
            'shape' : numpy.zeros(max_instances, dtype=numpy.long),
            'color' : numpy.zeros(max_instances, dtype=numpy.long),
            'pose' : numpy.zeros((max_instances, 4, 4)),
            'edges' : numpy.zeros((4, max_edges), dtype=numpy.long),
        }
        
        # break ----------------------------------------------------------------
        break_roadmap = Roadmap(env, empty_assembly, shape_ids, color_ids)
        break_planner = RoadmapPlanner(break_roadmap, full_state)
        
        def dump_scene(basename):
            scene = BrickScene()
            shape_ids = env.components['table_scene'].shape_ids
            color_ids = env.components['table_scene'].color_ids
            scene.set_assembly(full_assembly, shape_ids, color_ids)
            if not isinstance(basename, str):
                basename = basename.__class__.__name__
            scene.export_ldraw('./%s_%06i.mpd'%(basename, i))
        
        try:
            found_break = break_planner.plan(timeout=2)
            break_path = break_planner.greedy_path()
        except Exception as e:
            if error_handling == 'record':
                dump_scene(e)
                continue
            elif error_handling == 'pdb':
                import pdb
                pdb.set_trace()
            elif error_handling == 'raise':
                raise
            elif error_handling == 'continue':
                continue
        
        # get the break observations and actions -------------------------------
        (break_observations,
         break_actions) = break_roadmap.get_observation_action_seq(break_path)
        observation_seq.extend(break_observations)
        action_seq.extend(break_actions)
        
        # switch to make phase -------------------------------------------------
        action = env.no_op_action()
        action['phase'] = 1
        action_seq.append(action)
        _, reward, terminal, info = env.step(action)
        
        empty_state = env.get_state()
        
        # make -----------------------------------------------------------------
        make_roadmap = Roadmap(env, full_assembly, shape_ids, color_ids)
        make_planner = RoadmapPlanner(make_roadmap, empty_state)
        
        try:
            found_make = make_planner.plan(timeout=2)
            if found_make:
                make_path = make_planner.greedy_path()
            else:
                dump_scene('no_path_found')
                continue
        except Exception as e:
            if error_handling == 'record':
                dump_scene(e)
                continue
            elif error_handling == 'pdb':
                import pdb
                pdb.set_trace()
            elif error_handling == 'raise':
                raise
            elif error_handling == 'continue':
                continue
        
        complete_seqs += 1
        
        # get the make observations and actions --------------------------------
        (make_observations,
         make_actions) = make_roadmap.get_observation_action_seq(make_path)
        observation_seq.extend(make_observations)
        action_seq.extend(make_actions)
        
        # make the final action ------------------------------------------------
        action = env.no_op_action()
        action['phase'] = 2
        action_seq.append(action)
        
        total = i+1 - dataset_config.start
        iterate.set_description('Complete: %i/%i'%(complete_seqs, total))
        
        observation_seq = stack_numpy_hierarchies(*observation_seq)
        action_seq = stack_numpy_hierarchies(*action_seq)
        
        file_name = os.path.basename(dataset_paths['mpd'][i])
        file_name = file_name.replace('.mpd', '_1.npz')
        file_name = file_name.replace('.ldr', '_1.npz')
        path = os.path.join(episode_path, file_name)
        episode = {'observations':observation_seq, 'actions':action_seq}
        numpy.savez_compressed(path, episode=episode)

# dataset ======================================================================

class BreakAndMakeSequenceDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        subset,
        task='break_and_make',
    ):
        dataset_paths = get_dataset_paths(dataset, split, subset=subset)
        self.rollout_paths = dataset_paths['rollouts']
        self.task = task
    
    def __len__(self):
        return len(self.rollout_paths)
    
    def __getitem__(self, i):
        path = self.rollout_paths[i]
        data = numpy.load(path, allow_pickle=True)['rollout'].item()
        
        if self.task == 'break_only':
            i = numpy.where(data['actions']['phase'])[0]
            if len(i):
                data = index_hierarchy(data, slice(0, i[0]+1))
        
        elif self.task == 'break_and_count':
            # get all break actions
            i = numpy.where(data['actions']['phase'])[0]
            break_data = index_hierarchy(data, slice(0, i[0]+1))
            
            # get all insert actions
            # but modify them to have blank table observations and
            # appropriate hand observations
            j = numpy.where(data['actions']['insert_brick']['shape_id'])
            insert_data = index_hierarchy(data, j)
            insert_data['observations']['table_color_render'][:] = 102
            hand_frames = data['observations']['hand_color_render']
            insert_data['observations']['hand_color_render'][0] = 102
            insert_data['observations']['hand_color_render'][1:] = (
                hand_frames[j[0][:-1] + 1])
            
            # get the final phase switch action
            # and modify it as above
            final_data = index_hierarchy(data, [i[1]])
            final_data['observations']['table_color_render'][:] = 102
            final_data['observations']['hand_color_render'][0] = (
                hand_frames[[j[0][-1] + 1]])
            data = concatenate_numpy_hierarchies(
                break_data, insert_data, final_data)
        
        return data

'''
def break_and_make_data_collate(seqs):
    seqs, pad = auto_pad_stack_numpy_hierarchies(
        *seqs, pad_axis=0, stack_axis=1)
    return seqs, pad
'''

# build functions ==============================================================

def build_train_env(dataset_config):
    print('-'*80)
    print('Building train env')
    train_env = async_ltron(
        dataset_config.num_envs,
        break_and_make_env,
        task=dataset_config.task,
        dataset=dataset_config.dataset,
        split=dataset_config.train_split,
        subset=dataset_config.train_subset,
        table_image_width=dataset_config.table_image_width,
        table_image_height=dataset_config.table_image_height,
        hand_image_width=dataset_config.hand_image_width,
        hand_image_height=dataset_config.hand_image_height,
        table_map_width=dataset_config.table_map_width,
        table_map_height=dataset_config.table_map_height,
        hand_map_width=dataset_config.hand_map_width,
        hand_map_height=dataset_config.hand_map_height,
        max_episode_length=dataset_config.max_episode_length,
        train=True,
        check_collisions=True,
        randomize_viewpoint=dataset_config.randomize_viewpoint,
        randomize_colors=dataset_config.randomize_viewpoint,
    )
    
    return train_env


def build_test_env(dataset_config):
    print('-'*80)
    print('Building test env')
    test_env = async_ltron(
        dataset_config.num_envs,
        break_and_make_env,
        task=dataset_config.task,
        dataset=dataset_config.dataset,
        split=dataset_config.test_split,
        subset=dataset_config.test_subset,
        table_image_width=dataset_config.table_image_width,
        table_image_height=dataset_config.table_image_height,
        hand_image_width=dataset_config.hand_image_width,
        hand_image_height=dataset_config.hand_image_height,
        table_map_width=dataset_config.table_map_width,
        table_map_height=dataset_config.table_map_height,
        hand_map_width=dataset_config.hand_map_width,
        hand_map_height=dataset_config.hand_map_height,
        max_episode_length=dataset_config.max_episode_length,
        train=True,
        check_collisions=True,
        randomize_viewpoint=dataset_config.randomize_viewpoint,
        randomize_colors=dataset_config.randomize_viewpoint,
    )
    
    return test_env


def build_sequence_train_loader(config):
    print('-'*80)
    print('Building Break And Make Data Loader')
    dataset = BreakAndMakeSequenceDataset(
        config.dataset,
        config.train_split,
        config.train_subset,
        task=config.task,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.loader_workers,
        #collate_fn=break_and_make_data_collate,
        collate_fn=pad_stack_collate,
        shuffle=True,
    )
    
    return loader
