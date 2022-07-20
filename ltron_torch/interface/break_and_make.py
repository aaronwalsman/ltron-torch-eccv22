import math
import os
import copy

import numpy

import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

import tqdm

from conspiracy.log import SynchronousConsecutiveLog

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import len_hierarchy, index_hierarchy
from splendor.masks import color_index_to_byte
from ltron.gym.envs.break_and_make_env import BreakAndMakeEnv
from ltron.visualization.drawing import (
    write_text, block_upscale_image, draw_box)

from ltron_torch.models.padding import make_padding_mask
from ltron_torch.interface.utils import (
    bernoulli_or_max, categorical_or_max, categorical_or_max_2d)

class BreakAndMakeInterfaceConfig(Config):
    mode_loss_weight = 1.
    shape_loss_weight = 1.
    color_loss_weight = 1.
    table_spatial_loss_weight = 1.
    table_polarity_loss_weight = 1.
    hand_spatial_loss_weight = 1.
    hand_polarity_loss_weight = 1.
    
    disable_camera_losses = False
    
    factor_cursor_distribution = False
    spatial_loss_mode = 'cross_entropy'
    
    visualization_seqs = 10
    
    allow_snap_flip = False
    
    split = 'test'

class BreakAndMakeInterface:
    def __init__(self, config, model, optimizer):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        dummy_env = BreakAndMakeEnv(config)
        self.no_op_action = dummy_env.no_op_action()
    
    @staticmethod
    def make_train_log(checkpoint=None):
        train_log = SynchronousConsecutiveLog(
            'table_spatial_loss',
            'table_polarity_loss',
            'hand_spatial_loss',
            'hand_polarity_loss',
            'shape_loss',
            'color_loss',
            'mode_loss',
            'total_loss',
            compressed = True,
            capacity = 1024,
        )
        if checkpoint is not None:
            train_log.set_state(checkpoint)
        
        return train_log
    
    @staticmethod
    def make_test_log(checkpoint=None):
        test_log = SynchronousConsecutiveLog(
            'terminal_reward',
            compressed = False,
        )
        if checkpoint is not None:
            test_log.set_state(checkpoint)
        
        return test_log
    
    def observation_to_tensors(self, observation, pad):
        raise NotImplementedError
    
    def action_to_tensors(self, action, pad):
        
        # initialize y
        y = {}
        
        # get the device
        device = next(self.model.parameters()).device
        
        # build the mode tensor
        y_mode = numpy.zeros(action['phase'].shape, dtype=numpy.long)
        y_mode[action['phase'] == 1] = 0
        y_mode[action['phase'] == 2] = 1
        y_mode[action['disassembly'] == 1] = 2
        y_mode[action['pick_and_place'] == 1] = 3
        y_mode[action['pick_and_place'] == 2] = 4
        y_mode[action['rotate'] == 1] = 5
        y_mode[action['rotate'] == 2] = 6
        y_mode[action['rotate'] == 3] = 7
        y_mode[action['insert_brick']['shape'] != 0] = 8
        y_mode[action['table_viewpoint'] == 1] = 9
        y_mode[action['table_viewpoint'] == 2] = 10
        y_mode[action['table_viewpoint'] == 3] = 11
        y_mode[action['table_viewpoint'] == 4] = 12
        y_mode[action['table_viewpoint'] == 5] = 13
        y_mode[action['table_viewpoint'] == 6] = 14
        y_mode[action['table_viewpoint'] == 7] = 15
        y_mode[action['hand_viewpoint'] == 1] = 16
        y_mode[action['hand_viewpoint'] == 2] = 17
        y_mode[action['hand_viewpoint'] == 3] = 18
        y_mode[action['hand_viewpoint'] == 4] = 19
        y_mode[action['hand_viewpoint'] == 5] = 20
        y_mode[action['hand_viewpoint'] == 6] = 21
        y_mode[action['hand_viewpoint'] == 7] = 22
        last_mode = 22
        if self.config.factor_cursor_distribution:
            y_mode[action['table_cursor']['activate'] == 1] = 23
            y_mode[action['hand_cursor']['activate'] == 1] = 24
            last_mode = 24
        
        if self.config.allow_snap_flip:
            y_mode[action['rotate'] == 4] = last_mode + 1
            y_mode[action['rotate'] == 5] = last_mode + 2
            y_mode[action['rotate'] == 6] = last_mode + 3
            y_mode[action['rotate'] == 7] = last_mode + 4
        
        y['mode'] = torch.LongTensor(y_mode).to(device)
        
        for region in 'table', 'hand':
            # get the active sequence indices
            y['%s_i'%region] = torch.BoolTensor(
                action['%s_cursor'%region]['activate']).to(device)
            
            # get the click locations
            y['%s_yx'%region] = torch.LongTensor(
                action['%s_cursor'%region]['position']).to(device)
            
            # get the click polarity
            y['%s_p'%region] = torch.LongTensor(
                action['%s_cursor'%region]['polarity']).to(device)
        
        y['insert_i'] = y['mode'] == 8
        y['shape'] = torch.LongTensor(
            action['insert_brick']['shape']).to(device) - 1
        y['color'] = torch.LongTensor(
            action['insert_brick']['color']).to(device) - 1
        
        return y
    
    def loss(self, x, y, pad, log=None):
        
        # initialize loss
        loss = 0
        
        # get shape and device
        s, b, m = x['mode'].shape
        device = x['mode'].device
        
        # mode supervision
        pad = torch.LongTensor(pad).to(device)
        loss_mask = make_padding_mask(pad, (s,b), mask_value=True)
        mode_loss = cross_entropy(
            x['mode'].view(-1,m), y['mode'].view(-1), reduction='none')
        mode_loss = mode_loss.view(s,b) * loss_mask
        
        if self.config.disable_camera_losses:
            noncamera_actions = y['mode'] <= 8
            mode_loss = mode_loss * noncamera_actions
        
        mode_loss = mode_loss.mean() * self.config.mode_loss_weight
        loss = loss + mode_loss
        #print('new mode loss')
        #print('   ', mode_loss.cpu())
        #print('   ', torch.sum(x['mode']))
        #print('   ', torch.sum(y['mode']))
        #print(y['mode'])
        
        if log is not None:
            #log.add_scalar('train/mode_loss', mode_loss, clock[0])
            log.log(mode_loss=mode_loss)
        
        # table and hand supervision
        for region in 'table', 'hand':
            h, w = x[region].shape[-2:]
            i = y['%s_i'%region].view(-1)
            
            # if the output contains all elements of s,b
            if len(x[region].shape) == 5:
                x_region = x[region].reshape(s*b, 2, h*w)[i]
            # if a subset of s,b has already been selected from the output
            elif len(x[region].shape) == 4:
                x_region = x[region].reshape(-1, 2, h*w)
            
            if x_region.shape[0]:
                # spatial
                x_spatial = x_region[:,0]
                y_y = y['%s_yx'%region][:,:,0]
                y_x = y['%s_yx'%region][:,:,1]
                y_spatial = (y_y * w + y_x).view(-1)[i]
                if self.config.spatial_loss_mode == 'cross_entropy':
                    spatial_loss = cross_entropy(x_spatial, y_spatial)
                    spatial_loss = spatial_loss * getattr(
                        self.config, '%s_spatial_loss_weight'%region)
                elif self.config.spatial_loss_mode in ('bce', 'bse_plus'):
                    dense_y_spatial = torch.zeros_like(x_spatial)
                    y_y = y_y.view(-1)[i]
                    y_x = y_x.view(-1)[i]
                    ny = dense_y_spatial.shape[0]
                    dense_y_spatial[range(ny), (y_y * w + y_x)] = 1.
                    spatial_loss = binary_cross_entropy_with_logits(
                        x_spatial, dense_y_spatial)
                    spatial_loss = spatial_loss * getattr(
                        self.config, '%s_spatial_loss_weight'%region)
                
                loss = loss + spatial_loss
                #print('new %s spatial loss'%region)
                #print('   ', spatial_loss.cpu())
                #print('   ', torch.sum(x_spatial).cpu())
                #print('   ', torch.sum(y_spatial).cpu())
                
                # polarity
                x_p = x_region[:,1].view(-1, h*w)
                x_p = x_p[range(x_p.shape[0]), y_spatial]
                y_p = y['%s_p'%region].view(-1)[i].float()
                polarity_loss = binary_cross_entropy_with_logits(x_p, y_p)
                polarity_loss = polarity_loss * getattr(
                    self.config, '%s_polarity_loss_weight'%region)
                loss = loss + polarity_loss
                #print('new %s polarity loss'%region)
                #print('   ', polarity_loss.cpu())
                #print('   ', torch.sum(x_p).cpu())
                #print('   ', torch.sum(y_p).cpu())
                
                if log is not None:
                    #log.add_scalar(
                    #    'train/%s_spatial_loss'%region, spatial_loss, clock[0])
                    #log.add_scalar(
                    #    'train/%s_polarity_loss'%region,
                    #    polarity_loss,
                    #    clock[0],
                    #)
                    log.log(**{'%s_spatial_loss'%region:spatial_loss})
                    log.log(**{'%s_polarity_loss'%region:polarity_loss})
        
        # shape and color loss
        i = y['insert_i'].view(-1)
        for region in 'shape', 'color':
            n = x[region].shape[-1]
            if len(x[region].shape) == 3:
                x_region = x[region].view(s*b, n)[i]
            else:
                assert torch.sum(i) == x_region.shape[0]
                x_region = x[region]
            if x_region.shape[0]:
                y_region = y[region].view(-1)[i]
                region_loss = cross_entropy(x_region, y_region)
                region_loss = region_loss * getattr(
                    self.config, '%s_loss_weight'%region)
                loss = loss + region_loss
                
                if log is not None:
                    #log.add_scalar(
                    #   'train/%s_loss'%region, region_loss, clock[0])
                    log.log(**{'%s_loss'%region:region_loss})
        
        if log is not None:
            #log.add_scalar('train/total_loss', loss, clock[0])
            log.log(total_loss=loss)
        
        #print('new loss')
        #print(loss.cpu())
        return loss
    
    def tensor_to_actions(self, x, env, mode='sample'):
        s, b, num_modes = x['mode'].shape
        assert s == 1
        x_mode = x['mode'].view(b,-1)
        x_shape = x['shape'].view(b,-1)
        x_color = x['color'].view(b,-1)
        
        mode_action = categorical_or_max(x_mode, mode=mode).cpu().numpy()
        shape_action = categorical_or_max(x_shape, mode=mode).cpu().numpy()
        color_action = categorical_or_max(x_color, mode=mode).cpu().numpy()
        
        #region_yx = []
        #region_polarity = []
        #for region in 'table', 'hand':
        def get_cursor(region):
            h, w = x[region].shape[-2:]
            x_region = x[region].view(b, 2, h, w)
            
            # spatial
            x_spatial = x_region[:,0]
            region_y, region_x = categorical_or_max_2d(x_spatial, mode=mode)
            region_y = region_y.cpu().numpy()
            region_x = region_x.cpu().numpy()
            #region_yx.append((region_y, region_x))
            #region_yx = (region_y, region_x)
            
            # polarity
            x_polarity = x_region[:,1]
            x_polarity = x_polarity[range(b), region_y, region_x]
            polarity = bernoulli_or_max(x_polarity, mode=mode).cpu().numpy()
            #region_polarity.append(polarity)
            
            return region_y, region_x, polarity
        
        #(table_y, table_x), (hand_y, hand_x) = region_yx
        #table_polarity, hand_polarity = region_polarity
        table_y, table_x, table_polarity = get_cursor('table')
        hand_y, hand_x, hand_polarity = get_cursor('hand')
        
        actions = []
        for i in range(b):
            action = copy.deepcopy(self.no_op_action)
            mode = mode_action[i]
            if mode == 0:
                action['phase'] = 1
            elif mode == 1:
                action['phase'] = 2
            elif mode == 2: # disassembly
                action['disassembly'] = 1
                if not self.config.factor_cursor_distribution:
                    action['table_cursor']['activate'] = True
            elif mode == 3:
                action['pick_and_place'] = 1
                if not self.config.factor_cursor_distribution:
                    action['table_cursor']['activate'] = True
                    action['hand_cursor']['activate'] = True
            elif mode == 4:
                action['pick_and_place'] = 2
                if not self.config.factor_cursor_distribution:
                    action['hand_cursor']['activate'] = True
            elif mode == 5:
                action['rotate'] = 1
                if not self.config.factor_cursor_distribution:
                    action['table_cursor']['activate'] = True
            elif mode == 6:
                action['rotate'] = 2
                if not self.config.factor_cursor_distribution:
                    action['table_cursor']['activate'] = True
            elif mode == 7:
                action['rotate'] = 3
                if not self.config.factor_cursor_distribution:
                    action['table_cursor']['activate'] = True
            elif mode == 8:
                action['insert_brick']['shape'] = shape_action[i] + 1
                action['insert_brick']['color'] = color_action[i] + 1
            elif mode >= 9 and mode < 16:
                action['table_viewpoint'] = mode - 8
            elif mode >= 16 and mode < 23:
                action['hand_viewpoint'] = mode - 15
            
            last_mode = 22
            
            if self.config.factor_cursor_distribution:
                if mode == 23:
                    action['table_cursor']['activate'] = True
                if mode == 24:
                    action['hand_cursor']['activate'] = True
                last_mode = 24
            
            if mode == last_mode + 1:
                action['rotate'] = 4
            elif mode == last_mode + 2:
                action['rotate'] = 5
            elif mode == last_mode + 3:
                action['rotate'] = 6
            elif mode == last_mode + 4:
                action['rotate'] = 7
            
            if action['table_cursor']['activate']:
                action['table_cursor']['position'] = numpy.array(
                    (table_y[i], table_x[i]), dtype=numpy.long)
                action['table_cursor']['polarity'] = table_polarity[i]
            if action['hand_cursor']['activate']:
                action['hand_cursor']['position'] = numpy.array(
                    (hand_y[i], hand_x[i]), dtype=numpy.long)
                action['hand_cursor']['polarity'] = hand_polarity[i]
            
            actions.append(action)
        
        return actions
    
    def visualize_episodes(self, epoch, episodes, visualization_directory):
        num_seqs = min(
            self.config.visualization_seqs, episodes.num_seqs())
        for seq_id in tqdm.tqdm(range(num_seqs)):
            seq_path = os.path.join(
                visualization_directory, 'seq_%06i'%seq_id)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            
            seq = episodes.get_seq(seq_id)
            seq_len = len_hierarchy(seq)
            for frame_id in range(seq_len):
                frame_action = index_hierarchy(seq['action'], frame_id)
                frame_observation = index_hierarchy(
                    seq['observation'], frame_id)
                
                magenta = numpy.array([255,0,255]).reshape(1,1,3)
                cyan = numpy.array([0,255,255]).reshape(1,1,3)
                
                def draw_cursor(region):
                    frames = seq['observation']['%s_color_render'%region]
                    frame = frames[frame_id]
                    h, w = frame.shape[:2]
                    if frame_action['%s_cursor'%region]['activate'] == 1:
                        activation = index_hierarchy(
                            seq['activations'][region], frame_id)
                        
                        # softmax
                        location = activation[0].astype(numpy.float)
                        location = numpy.exp(location)
                        norm = numpy.sum(location)
                        location /= norm
                        
                        # upsample
                        location = block_upscale_image(
                            location, h, w).reshape(h, w, 1)
                        
                        polarity = activation[1].astype(numpy.float)
                        polarity = 1. / (1. + math.e ** -polarity)
                        polarity = block_upscale_image(polarity, h, w)
                        polarity = polarity.reshape(h, w, 1)
                        color = magenta * (1.-polarity) + cyan * polarity
                        
                        #frame = (
                        #    frame * 0.5 * (1. - location) +
                        #    color * location).astype(numpy.uint8)
                        
                        y, x = frame_action['%s_cursor'%region]['position']
                        y, x = int(y), int(x)
                        p = frame_action['%s_cursor'%region]['polarity']
                        p = int(p)
                        #draw_box(
                        #    frame,
                        #    x*4, y*4, (x+1)*4-1, (y+1)*4-1,
                        #    (255*(1-p), 0, 255*p),
                        #)
                    
                    else:
                        y, x = frame_observation['%s_cursor'%region]['position']
                        y, x = int(y), int(x)
                        p = frame_observation['%s_cursor'%region]['polarity']
                        p = int(p)
                        #draw_box(
                        #    frame,
                        #    x*4, y*4, (x+1)*4-1, (y+1)*4-1,
                        #    (255*(1-p), 0, 255*p),
                        #)
                    
                    pos_snaps = seq['observation']['%s_pos_snap_render'%region]
                    pos_snap = pos_snaps[frame_id,:,:,0]
                    pos_snap = color_index_to_byte(pos_snap)
                    pos_snap = block_upscale_image(pos_snap, h, w)
                    draw_box(
                        pos_snap,
                        x*4, y*4, (x+1)*4-1, (y+1)*4-1, (255, 255, 255),
                    )
                    
                    neg_snaps = seq['observation']['%s_neg_snap_render'%region]
                    neg_snap = neg_snaps[frame_id,:,:,0]
                    neg_snap = color_index_to_byte(neg_snap)
                    neg_snap = block_upscale_image(neg_snap, h, w)
                    draw_box(
                        neg_snap,
                        x*4, y*4, (x+1)*4-1, (y+1)*4-1, (255, 255, 255),
                    )
                    
                    return frame, pos_snap, neg_snap
                        
                table_frame, table_pos, table_neg = draw_cursor('table')
                hand_frame, hand_pos, hand_neg = draw_cursor('hand')
                
                th, tw = table_frame.shape[:2]
                hh, hw = hand_frame.shape[:2]
                w = tw + hw
                #joined_image = numpy.zeros((th*3, w, 3), dtype=numpy.uint8)
                joined_image = numpy.zeros((th, w, 3), dtype=numpy.uint8)
                joined_image[:th,:tw] = table_frame
                joined_image[th-hh:th,tw:] = hand_frame
                
                #joined_image[th:th*2,:tw] = table_pos
                #joined_image[th*2-hh:th*2,tw:] = hand_pos
                
                #joined_image[th*2:th*3,:tw] = table_neg
                #joined_image[th*3-hh:th*3,tw:] = hand_neg
                
                mode_string = []
                if frame_action['insert_brick']['shape']:
                    shape = frame_action['insert_brick']['shape']
                    color = frame_action['insert_brick']['color']
                    mode_string.append('Insert Brick [%i] [%i]'%(shape, color))
                elif frame_action['pick_and_place'] == 1:
                    mode_string.append('Pick And Place')
                elif frame_action['pick_and_place'] == 2:
                    mode_string.append('Pick And Place ORIGIN')
                elif frame_action['rotate']:
                    mode_string.append('Rotate [%i]'%frame_action['rotate'])
                elif frame_action['disassembly']:
                    mode_string.append('Disassembly')
                elif frame_action['table_viewpoint']:
                    mode_string.append('Table Viewpoint [%i]'%(
                        frame_action['table_viewpoint']))
                elif frame_action['hand_viewpoint']:
                    mode_string.append('Hand Viewpoint [%i]'%(
                        frame_action['hand_viewpoint']))
                elif frame_action['phase']:
                    mode_string.append('Phase [%i]'%frame_action['phase'])
                elif frame_action['table_cursor']['activate']:
                    mode_string.append('Table Cursor')
                elif frame_action['hand_cursor']['activate']:
                    mode_string.append('Hand Cursor')
                else:
                    mode_string.append('UNKNOWN ACTION')
                
                if frame_id:
                    reward = seq['reward'][frame_id-1]
                else:
                    reward = 0.
                mode_string.append('Reward: %.04f'%reward)
                
                mode_string = '\n'.join(mode_string)
                
                try:
                    joined_image = write_text(joined_image, mode_string)
                except OSError:
                    pass
                
                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                save_image(joined_image, frame_path)
    
    def test_episodes(self, episodes, test_log):
        disassembly_n = 0
        disassembly_correct = 0
        nonconsecutive_disassembly_n = 0
        pick_and_place_n = 0
        pick_and_place_correct = 0
        pick_and_place_one_correct = 0
        pick_and_place_both_correct = 0
        rotation_n = 0
        rotation_correct = 0
        phase_n = 0
        phase_correct = 0
        
        brick_correct = 0
        end_in_phase_0 = 0
        
        for i in range(episodes.num_seqs()):
            seq = episodes.get_seq(i)
            seq_len = len_hierarchy(seq)
            disassembly_failed = False
            
            # are the number of produced bricks correct?
            first_step = index_hierarchy(seq, 0)
            initial_table_assembly = (
                first_step['observation']['initial_table_assembly'])
            correct_shape_colors = set()
            num_bricks = initial_table_assembly['shape'].shape[0]
            for k in range(num_bricks):
                shape = initial_table_assembly['shape'][k]
                color = initial_table_assembly['color'][k]
                if shape:
                    correct_shape_colors.add((shape, color))
            
            last_step = index_hierarchy(seq, seq_len-1)
            last_assembly = (
                last_step['observation']['table_assembly'])
            last_shape_colors = set()
            num_bricks = last_assembly['shape'].shape[0]
            for k in range(num_bricks):
                shape = last_assembly['shape'][k]
                color = last_assembly['color'][k]
                if shape:
                    last_shape_colors.add((shape, color))
            
            brick_correct += (correct_shape_colors == last_shape_colors)
            
            # what phase did the episode end in?
            last_phase = last_step['observation']['phase']
            if last_phase == 0:
                end_in_phase_0 += 1
            
            # make the good edges for pick and place checking later
            table_edges = initial_table_assembly['edges']
            n_edges = table_edges.shape[1]
            good_edges = set()
            good_snaps = set()
            for k in range(n_edges):
                a_i, b_i, a_s, b_s = table_edges[:,k]
                if a_i == 0 or b_i == 0:
                    continue
                a_shape = initial_table_assembly['shape'][a_i]
                a_color = initial_table_assembly['color'][a_i]
                b_shape = initial_table_assembly['shape'][b_i]
                b_color = initial_table_assembly['color'][b_i]
                good_edges.add(
                    (a_shape, a_color, a_s, b_shape, b_color, b_s))
                good_snaps.add((a_shape, a_color, a_s))
                good_snaps.add((b_shape, b_color, b_s))
            
            for j in range(seq_len):
                step = index_hierarchy(seq, j)
                if step['action']['disassembly']:
                    # was this a good disassembly action?
                    disassembly_n += 1
                    if not disassembly_failed:
                        nonconsecutive_disassembly_n += 1
                    if j != seq_len-1:
                        next_step = index_hierarchy(seq, j+1)
                        instances_now = numpy.sum(
                            step['observation']['table_assembly']['shape'] != 0)
                        instances_next = numpy.sum(
                            next_step['observation']['table_assembly']['shape']
                            != 0)
                        if instances_next < instances_now:
                            disassembly_correct += 1
                            disassembly_failed=False
                        else:
                            disassembly_failed=True
                    else:
                        disassembly_failed = True
                else:
                    disassembly_failed = False
                
                if step['action']['pick_and_place'] == 1:
                    # was this a good pick and place action?
                    pick_and_place_n += 1
                    hand_y, hand_x = step['action']['hand_cursor']['position']
                    hand_p = step['action']['hand_cursor']['polarity']
                    table_y, table_x = (
                        step['action']['table_cursor']['position'])
                    table_p = step['action']['table_cursor']['polarity']
                    if hand_p:
                        hand_snap_map = (
                            step['observation']['hand_pos_snap_render'])
                    else:
                        hand_snap_map = (
                            step['observation']['hand_neg_snap_render'])
                    hand_instance_id, hand_snap_id = hand_snap_map[
                        hand_y, hand_x]
                    hand_assembly = step['observation']['hand_assembly']
                    hand_shape = hand_assembly['shape'][hand_instance_id]
                    hand_color = hand_assembly['color'][hand_instance_id]
                    
                    if table_p:
                        table_snap_map = (
                            step['observation']['table_pos_snap_render'])
                    else:
                        table_snap_map = (
                            step['observation']['table_neg_snap_render'])
                    table_instance_id, table_snap_id = table_snap_map[
                        table_y, table_x]
                    table_assembly = step['observation']['table_assembly']
                    table_shape = table_assembly['shape'][table_instance_id]
                    table_color = table_assembly['color'][table_instance_id]
                    
                    edge_a = (
                        hand_shape, hand_color, hand_snap_id,
                        table_shape, table_color, table_snap_id,
                    )
                    edge_b = (
                        table_shape, table_color, table_snap_id,
                        hand_shape, hand_color, hand_snap_id,
                    )
                    
                    if edge_a in good_edges or edge_b in good_edges:
                        pick_and_place_correct += 1
                    
                    hand_snap = (hand_shape, hand_color, hand_snap_id)
                    table_snap = (table_shape, table_color, table_snap_id)
                    if hand_snap in good_snaps or table_snap in good_snaps:
                        pick_and_place_one_correct += 1
                    
                    if hand_snap in good_snaps and table_snap in good_snaps:
                        pick_and_place_both_correct += 1
                
                if step['action']['rotate']:
                    # was this a good rotate action?
                    # Nevermind, this is hard... how do you make sure you're
                    # comparing the right thing if both bricks have the same
                    # pose and shape?  Anyway, we have enough to do to figure
                    # out better pick and place anyway, so let's focus on that.
                    # Plus, once we figure that out, this will be easier to
                    # analyze anyway.
                    #if j != seq_len - 1:
                    #    initial_shape_1 = initial_table_assembly['shape'][1]
                    #    initial_color_1 = initial_table_assembly['color'][1]
                    #    initial_pose_1 =  initial_table_assembly['pose'][1]
                    #    initial_shape_2 = initial_table_assembly['shape'][2]
                    #    initial_color_2 = initial_table_assembly['color'][2]
                    #    initial_pose_2 =  initial_table_assembly['pose'][2]
                    #    
                    #    next_step = index_hierarchy(seq, j+1)
                    #    next_table_assembly = next_step['table_assembly']
                    #    next_shape_1 = next_table_assembly['shape'][1]
                    #    next_color_1 = next_table_assembly['color'][1]
                    #    next_pose_1 =  next_table_assembly['pose'][1]
                    #    next_shape_2 = next_table_assembly['shape'][2]
                    #    next_color_2 = next_table_assembly['color'][2]
                    #    next_pose_2 =  next_table_assembly['pose'][2]
                    pass
                        
        
        if episodes.num_seqs():
            print('Correct Bricks: %.04f'%(brick_correct / episodes.num_seqs()))
            print('Switched to Disassembly: %.04f'%(
                1. - end_in_phase_0 / episodes.num_seqs()))
        if disassembly_n:
            print('Disassembly: %.04f'%(disassembly_correct / disassembly_n))
        if nonconsecutive_disassembly_n:
            print('Nonconsecutive Disassembly: %.04f'%(
                disassembly_correct / nonconsecutive_disassembly_n))
        if pick_and_place_n:
            print('Pick and Place: %.04f'%(
                pick_and_place_correct / pick_and_place_n))
            print('Pick and Place One Correct: %.04f'%(
                pick_and_place_one_correct / pick_and_place_n))
            print('Pick and Place Both Correct: %.04f'%(
                pick_and_place_both_correct / pick_and_place_n))
    
    def numpy_activations(self, x):
        a = {
            key:value.cpu().numpy().squeeze(axis=0)
            for key, value in x.items()
        }
        return a
