import os

import numpy

import torch
from torch.nn.functional import cross_entropy

import tqdm

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.envs.blocks_env import BlocksEnv
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import make_padding_mask
from ltron_torch.interface.utils import (
    categorical_or_max, categorical_or_max_2d)

class BlocksInterfaceConfig(Config):
    mode_loss_weight = 1.
    shape_loss_weight = 1.
    color_loss_weight = 1.
    table_loss_weight = 1.
    hand_loss_weight = 1.
    
    visualization_seqs = 10

class BlocksInterface:
    def __init__(self, model, config):
        self.model = model
        self.config = config
    
    def observation_to_tensors(self, observation, pad):
        raise NotImplementedError
    
    def loss(self, x, pad, y, log=None, clock=None):
        s, b = x['mode'].shape[:2]
        device = x['mode'].device
        
        # make the padding mask
        pad = torch.LongTensor(pad).to(device)
        loss_mask = make_padding_mask(pad, (s,b), mask_value=True)
        
        # mode, shape and color supervision
        y_mode = torch.LongTensor(y['mode']).to(x['mode'].device)
        num_modes = x['mode'].shape[-1]
        mode_loss = cross_entropy(
            x['mode'].view(-1,num_modes), y_mode.view(-1), reduction='none')
        mode_loss = mode_loss.view(s,b) * loss_mask
        mode_loss = mode_loss.mean() * self.config.mode_loss_weight
        
        # hand and table supervision
        table_t = ((y_mode == 0) | (y_mode == 1)).view(-1)
        hand_t = (y_mode == 1).view(-1)
        spatial_losses = []
        for region, t in ('table', table_t), ('hand', hand_t):
            h, w = x[region].shape[-2:]
            x[region] = x[region].view(s*b, h*w)[t]
            y_y = torch.LongTensor(y[region + '_cursor'][:,:,0])
            y_x = torch.LongTensor(y[region + '_cursor'][:,:,1])
            y_region = (y_y * w + y_x).view(-1)[t].to(device)
            loss = cross_entropy(x[region], y_region)
            loss = loss * getattr(self.config, '%s_loss_weight'%region)
            spatial_losses.append(loss)
        
        table_loss, hand_loss = spatial_losses
        
        # shape and color loss
        shape_color_t = (y_mode == 2).view(-1)
        shape_color_losses = []
        #for x_region, y_region, loss_weight in(
        #    (x['shape'], y['shape'], self.config.shape_loss_weight),
        #    (x['color'], y['color'], self.config.color_loss_weight),
        #):
        for region in 'shape', 'color':
            num_x = x[region].shape[-1]
            x_region = x[region].view(s*b, num_x)[shape_color_t]
            y_region = torch.LongTensor(y[region])
            y_region = y_region.view(-1)[shape_color_t].to(device)
            loss = cross_entropy(x_region, y_region)
            loss = loss * getattr(self.config, '%s_loss_weight'%region)
            shape_color_losses.append(loss)
        
        shape_loss, color_loss = shape_color_losses
        
        total_loss = (
            mode_loss + table_loss + hand_loss + shape_loss + color_loss)
        
        if log is not None:
            log.add_scalar('train/mode_loss', mode_loss, clock[0])
            log.add_scalar('train/table_loss', table_loss, clock[0])
            log.add_scalar('train/hand_loss', hand_loss, clock[0])
            log.add_scalar('train/shape_loss', shape_loss, clock[0])
            log.add_scalar('train/color_loss', color_loss, clock[0])
            log.add_scalar('train/total_loss', total_loss, clock[0])
        
        return total_loss

    def tensor_to_actions(self, x, env, mode='sample'):
        s, b, num_modes = x['mode'].shape
        assert s == 1
        x_mode = x['mode'].view(b,-1)
        x_shape = x['shape'].view(b,-1)
        x_color = x['color'].view(b,-1)
        
        mode_action = categorical_or_max(x_mode, mode=mode).cpu().numpy()
        shape_action = categorical_or_max(x_shape, mode=mode).cpu().numpy()
        color_action = categorical_or_max(x_color, mode=mode).cpu().numpy()
        
        region_yx = []
        for region in 'table', 'hand':
            #s, b, c, h, w = x[region].shape
            h, w = x[region].shape[-2:]
            x_region = x[region].view(b, 1, h, w)
            region_y, region_x = categorical_or_max_2d(x_region, mode=mode)
            region_y = region_y.cpu().numpy()
            region_x = region_x.cpu().numpy()
            region_yx.append((region_y, region_x))
        (table_y, table_x), (hand_y, hand_x) = region_yx
        
        actions = []
        for i in range(b):
            action = BlocksEnv.no_op_action()
            action['mode'] = mode_action[i]
            action['shape'] = shape_action[i]
            action['color'] = color_action[i]
            action['table_cursor'] = numpy.array([table_y[i], table_x[i]])
            action['hand_cursor'] = numpy.array([hand_y[i], hand_x[i]])
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
            table_frames = seq['observation']['table_render']
            hand_frames = seq['observation']['hand_render']
            for frame_id in range(seq_len):
                table_frame = table_frames[frame_id]
                hand_frame = hand_frames[frame_id]
                th, tw = table_frame.shape[:2]
                hh, hw = hand_frame.shape[:2]
                w = tw + hw
                joined_image = numpy.zeros((th, w, 3), dtype=numpy.uint8)
                joined_image[:,:tw] = table_frame
                joined_image[th-hh:,tw:] = hand_frame

                frame_action = index_hierarchy(seq['action'], frame_id)
                frame_mode = int(frame_action['mode'])
                frame_shape_id = int(frame_action['shape'])
                frame_color_id = int(frame_action['color'])
                ty, tx = frame_action['table_cursor']
                ty = int(ty)
                tx = int(tx)
                hy, hx = frame_action['hand_cursor']
                hy = int(hy)
                hx = int(hx)

                joined_image[ty*4:(ty+1)*4, tx*4:(tx+1)*4] = (0,0,0)
                yy = th - hh
                joined_image[
                    yy+hy*4:yy+(hy+1)*4, tw+(hx)*4:tw+(hx+1)*4] = (0,0,0)

                mode_string = 'Mode: %s'%([
                    'disassemble',
                    'place',
                    'pick-up',
                    'make',
                    'end',
                    'no-op'][frame_mode])
                shape_string = 'Shape: %s'%str(
                    self.config.block_shapes[frame_shape_id])
                color_string = 'Color: %s'%str(
                    self.config.block_colors[frame_shape_id])
                lines = (mode_string, shape_string, color_string)
                joined_image = write_text(joined_image, '\n'.join(lines))

                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                save_image(joined_image, frame_path)
