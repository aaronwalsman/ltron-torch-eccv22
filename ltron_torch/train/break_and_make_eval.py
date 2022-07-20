import random

import numpy

import torch

from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import BreakAndMakeEnv
from ltron.evaluation import precision_recall, f1
from ltron.score import score_assemblies, edit_distance
from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.bricks.brick_scene import BrickScene

from ltron_torch.models.hand_table_transformer import (
    HandTableTransformerConfig, HandTableTransformer)
from ltron_torch.models.stubnet_transformer import (
    StubnetTransformerConfig, StubnetTransformer)
from ltron_torch.models.hand_table_lstm import (
    HandTableLSTMConfig, HandTableLSTM)
from ltron_torch.interface.break_and_make_hand_table_transformer import (
    BreakAndMakeHandTableTransformerInterface)
from ltron_torch.interface.break_and_make_stubnet_transformer import (
    BreakAndMakeStubnetTransformerInterface)
from ltron_torch.interface.break_and_make_hand_table_lstm import (
    BreakAndMakeHandTableLSTMInterface)

from ltron_torch.train.break_and_make_bc import BreakAndMakeBCConfig
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, rollout_epoch,
)

class EvalConfig(BreakAndMakeBCConfig):
    evaluation_subset = None
    dump_mpd = False

def break_and_make_eval(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = EvalConfig.from_commandline()
        
    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if config.factor_cursor_distribution:
        config.num_modes = 25
    else:
        config.num_modes = 23

    if config.allow_snap_flip:
        config.num_modes += 4
    
    config.num_test_envs = 1
    
    device = torch.device(config.device)
    
    dataset_info = get_dataset_info(config.dataset)
    part_names = {value:key for key,value in dataset_info['shape_ids'].items()}
    test_len = len(get_dataset_paths(config.dataset, config.test_split)['mpd'])
    
    if config.evaluation_subset is None:
        config.test_episodes_per_epoch = test_len
    else:
        config.test_episodes_per_epoch = config.evaluation_subset
    
    print('-'*80)
    print('Loading Checkpoint')
    checkpoint = torch.load(config.load_checkpoint)
    model_checkpoint = checkpoint['model']
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
    print('Building Interface (%s)'%config.model)
    if config.model == 'transformer':
        interface = BreakAndMakeHandTableTransformerInterface(
            config, model, None)
    elif config.model == 'stubnet':
        interface = BreakAndMakeStubnetTransformerInterface(
            config, model, None)
    elif config.model == 'lstm':
        interface = BreakAndMakeHandTableLSTMInterface(
            config, model, None)
    
    print('-'*80)
    print('Building Test Env')
    test_config = BreakAndMakeBCConfig.translate(config, split='test_split')
    #if config.async_ltron:
    #    vector_ltron = async_ltron
    #else:
    vector_ltron = sync_ltron
    test_env = vector_ltron(
        config.num_test_envs,
        BreakAndMakeEnv,
        test_config,
        print_traceback=True,
    )
    
    test_env.seed(config.seed)
    
    episodes = rollout_epoch(
        0,
        config,
        test_env,
        model,
        interface,
        'test',
        True,
        False,
    )
    
    brick_f1s = []
    edit_dists = []
    edge_f1s = []
    assembly_scores = []
    
    if config.dump_mpd:
        scene = BrickScene()
    
    for i in range(episodes.num_finished_seqs()):
        seq = episodes.get_seq(i)
        seq_len = len_hierarchy(seq)
        
        first_step = index_hierarchy(seq, 0)
        initial_table_assembly = (
            first_step['observation']['initial_table_assembly'])
        num_bricks = numpy.sum(initial_table_assembly['shape'] != 0)
        
        last_step = index_hierarchy(seq, seq_len-1)
        final_table_assembly = (
            last_step['observation']['table_assembly'])
        last_phase = last_step['observation']['phase']
        
        # brick F1
        if last_phase == 0:
            brick_f1 = 0.
        else:
            initial_shape = initial_table_assembly['shape']
            initial_color = initial_table_assembly['color']
            final_shape = final_table_assembly['shape']
            final_color = final_table_assembly['color']
            def get_bricks(shapes, colors):
                #return set((s, c) for s, c in zip(shapes, colors) if s != 0)
                out = {}
                for s, c in zip(shapes, colors):
                    if s == 0:
                        continue
                    if (s,c) not in out:
                        out[s,c] = 0
                    out[s,c] += 1
                return out
            
            initial_bricks = get_bricks(initial_shape, initial_color)
            final_bricks = get_bricks(final_shape, final_color)
            
            def multiset_intersect(a, b):
                result = {}
                keys = set()
                keys |= a.keys()
                keys |= b.keys()
                for key in keys:
                    if key in a and key in b:
                        result[key] = min(a[key], b[key])
                
                return result
            
            def multiset_subtract(a, b):
                result = {}
                for key in a:
                    if key in b:
                        result[key] = max(0, a[key] - b[key])
                    else:
                        result[key] = a[key]
                return result
            
            def multiset_len(a):
                return sum(a.values())
            
            tp = multiset_intersect(initial_bricks, final_bricks)
            fp = multiset_subtract(final_bricks, initial_bricks)
            fn = multiset_subtract(initial_bricks, final_bricks)
            #tp = len(initial_bricks & final_bricks)
            #fp = len(final_bricks - initial_bricks)
            #fn = len(initial_bricks - final_bricks)
            p, r = precision_recall(
                multiset_len(tp), multiset_len(fp), multiset_len(fn))
            brick_f1 = f1(p, r)
        
        brick_f1s.append(brick_f1)
        
        # edit distance
        if last_phase == 0:
            #assembly_edit_distance = 0
            assembly_edit_distance = 2 * num_bricks
            initial_to_final_matching = {}
        else:
            assembly_edit_distance, initial_to_final_matching = edit_distance(
                initial_table_assembly,
                final_table_assembly,
                part_names,
                miss_a_penalty=2,
                miss_b_penalty=1,
            )
        edit_dists.append(assembly_edit_distance)
        all_initial_instances = numpy.where(initial_table_assembly['shape'])[0]
        all_final_instances = numpy.where(final_table_assembly['shape'])[0]
        max_final_instance = max(all_final_instances, default=0)
        for j in all_initial_instances:
            if j not in initial_to_final_matching:
                initial_to_final_matching[j] = j + max_final_instance
        
        # edge f1
        if last_phase == 0:
            edge_f1 = 0
        else:
            def simplified_edges(a):
                return set(
                    (a_i, b_i) for a_i, b_i, a_s, b_s in a['edges'].T
                    if a_i != 0 and b_i != 0 and a_i < b_i
                )
            initial_edges = simplified_edges(initial_table_assembly)
            remapped_edges = {
                (initial_to_final_matching[a], initial_to_final_matching[b])
                for a, b in initial_edges
            }
            remapped_edges = {
                (a, b) if a < b else (b, a) for a, b in remapped_edges
            }
            final_edges = simplified_edges(final_table_assembly)
            tp = len(remapped_edges & final_edges)
            fp = len(final_edges - remapped_edges)
            fn = len(remapped_edges - final_edges)
            p, r = precision_recall(tp, fp, fn)
            edge_f1 = f1(p,r)
        edge_f1s.append(edge_f1)
        
        if last_phase == 0:
            assembly_score = 0
        else:
            assembly_score, _ = score_assemblies(
                initial_table_assembly, final_table_assembly, part_names)
        assembly_scores.append(assembly_score)
        
        if config.dump_mpd:
            scene.clear_instances()
            scene.import_assembly(
                final_table_assembly,
                dataset_info['shape_ids'],
                dataset_info['color_ids'],
            )
            scene.export_ldraw('./predicted_%06i.mpd'%i)
            
            scene.clear_instances()
            scene.import_assembly(
                initial_table_assembly,
                dataset_info['shape_ids'],
                dataset_info['color_ids'],
            )
            scene.export_ldraw('./target_%06i.mpd'%i)
    
    print('Average Brick F1: %f'%(sum(brick_f1s) / len(brick_f1s)))
    print('Average Edit Distance: %f'%(sum(edit_dists) / len(edit_dists)))
    print('Average Edge F1: %f'%(sum(edge_f1s) / len(edge_f1s)))
    print('Average Assembly Score: %f'%(
        sum(assembly_scores) / len(assembly_scores)))
