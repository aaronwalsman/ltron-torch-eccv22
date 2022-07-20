import os

import numpy

import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

import tqdm

from splendor.image import save_image

from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.visualization.drawing import write_text

from ltron_torch.gym_tensor import default_image_transform
from ltron_torch.models.padding import make_padding_mask
from ltron_torch.interface.utils import (
    categorical_or_max, categorical_or_max_2d)
from ltron_torch.interface.break_and_make import BreakAndMakeInterface

class BreakAndMakeHandTableLSTMInterface(BreakAndMakeInterface):
    def observation_to_tensors(self, observation, action, pad):
        # get the device
        device = next(self.model.parameters()).device
        
        output = {}
        for region in 'table', 'hand':
            frames = observation['%s_color_render'%region]
            s, b, h, w, c = frames.shape
            frames = frames.reshape(s*b, h, w, c)
            frames = [default_image_transform(frame) for frame in frames]
            frames = torch.stack(frames)
            frames = frames.view(s, b, c, h, w).to(device)
            output['x_%s'%region] = frames
        
        phase = torch.LongTensor(observation['phase']).to(device)
        output['r'] = phase
        
        return output
    
    def numpy_activations(self, x):
        a = {
            key:value.cpu().numpy().squeeze(axis=0)
            for key, value in x.items()
            if key != 'memory'
        }
        return a
    
    #def loss(self, x, pad, y, log=None, clock=None):
    #    return blocks_loss(self.config, x, pad, y, log, clock)
    #    '''
    #    y_mode = torch.zeros(s, b, dtype=torch.long)
    #    for j in range(b):
    #        for i in range(pad[j]):
    #            if y['disassembly'][i,j]:
    #                y_mode[i,j] = 0
    #                continue
    #            elif y['pick_and_place'][i,j]:
    #                pick_and_place = y['pick_and_place'][i,j]
    #                y_mode[i,j] = pick_and_place
    #                continue
    #            elif y['rotate'][i,j]:
    #                y_mode[i,j] = y['rotate'][i,j] + 2
    #                continue
    #            elif y['workspace_viewpoint'][i,j]:
    #                y_mode[i,j] = (
    #                    y['workspace_viewpoint'][i,j] + 5)
    #                continue
    #            elif y['handspace_viewpoint'][i,j]:
    #                y_mode[i,j] = (
    #                    y['handspace_viewpoint'][i,j] + 12)
    #                continue
    #            elif y['phase'][i,j]:
    #                y_mode[i,j] = y['phase_switch'][i,j] + 19
    #                continue
    #            elif y['insert_brick']['class_id'][i,j]:
    #                y_mode[i,j] = 22
    #                continue
    #            else:
    #                import pdb
    #                pdb.set_trace()
    #    
    #    mode_loss = cross_entropy(
    #        x['mode'].view(-1,23),
    #        y_mode.view(-1).cuda(),
    #        reduction='none',
    #    ).view(s,b)
    #    mode_loss = mode_loss * loss_mask
    #    mode_loss = torch.sum(mode_loss) / (s*b)
    #    
    #    if 'class' in x:
    #        y_class = torch.LongTensor(
    #            y['insert_brick']['class_id']).cuda()
    #        class_loss = cross_entropy(
    #            x['class'].view(s*b, -1), y_class.view(-1), reduction='none')
    #        insert_mask = (y_class.view(-1) != 0)
    #        class_loss = class_loss * insert_mask
    #        class_loss = torch.sum(class_loss) / (s*b)
    #        
    #        p_class = torch.softmax(x['class'], dim=-1).view(s*b,-1)
    #        correct_class = y_class.view(-1)
    #        p_correct_class = p_class[range(s*b), correct_class]
    #        p_correct_class = p_correct_class[insert_mask]
    #        avg_p_correct_class = (
    #            torch.sum(p_correct_class)/p_correct_class.shape[0])
    #    else:
    #        class_loss = 0.
    #        avg_p_correct_class = 0.
    #    
    #    if 'color' in x:
    #        y_color = torch.LongTensor(
    #            y['insert_brick']['color_id']).cuda()
    #        color_loss = cross_entropy(
    #            x['color'].view(s*b, -1), y_color.view(-1), reduction='none')
    #        color_loss = color_loss * insert_mask
    #        color_loss = torch.sum(color_loss) / (s*b)
    #        
    #        p_color = torch.softmax(x['color'], dim=-1).view(s*b,-1)
    #        correct_color = y_color.view(-1)
    #        p_correct_color = p_color[range(s*b), correct_color]
    #        p_correct_color = p_correct_color[insert_mask]
    #        avg_p_correct_color = (
    #            torch.sum(p_correct_color)/p_correct_color.shape[0])
    #    else:
    #        color_loss = 0.
    #        avg_p_correct_color = 0.
    #    
    #    if log is not None:
    #        log.add_scalar('train/mode_loss', mode_loss, clock[0])
    #        log.add_scalar('train/class_loss', class_loss, clock[0])
    #        log.add_scalar('train/color_loss', color_loss, clock[0])
    #        log.add_scalar('train/correct_class', avg_p_correct_class, clock[0])
    #        log.add_scalar('train/correct_color', avg_p_correct_color, clock[0])
    #    
    #    position_losses = []
    #    polarity_losses = []
    #    for space in 'workspace', 'handspace':
    #        if space not in x:
    #            position_losses.append(0.)
    #            polarity_losses.append(0.)
    #            continue
    #        
    #        x_space = x[space]
    #        h, w = x_space.shape[-2:]

    #        position = torch.LongTensor(
    #            y['%s_cursor'%space]['position']).view(s*b, 2)
    #        if False:
    #            x_position = x_space[:,:,0].view(s*b, h, w)
    #            y_position = torch.zeros_like(x_position)
    #            y_position[range(s*b), position[:,0], position[:,1]] = 1
    #            weight = (w * h) // 32
    #            dense_position_loss = binary_cross_entropy_with_logits(
    #                x_position,
    #                y_position,
    #                pos_weight=torch.FloatTensor([weight]).cuda(),
    #                reduction='none',
    #            )
    #            dense_position_loss = dense_position_loss.view(s, b, h*w)
    #            position_loss = torch.mean(dense_position_loss, axis=-1) * 15

    #        else:
    #            x_position = x_space[:,:,0].view(s*b, h*w)
    #            y_position = (position[:,0]*w + position[:,1]).cuda()
    #            y_position = y_position.view(-1)
    #            position_loss = cross_entropy(
    #                x_position, y_position, reduction='none').view(s,b)

    #        if space == 'workspace':
    #            mode_mask = (y_mode <= 1) | ((y_mode >= 3) & (y_mode <= 5))
    #        elif space == 'handspace':
    #            mode_mask = (y_mode == 1) | (y_mode == 2)
    #        
    #        position_mask = loss_mask * mode_mask.cuda()
    #        position_loss = position_loss * position_mask
    #        position_loss = torch.sum(position_loss) / (s*b)

    #        x_polarity = x_space[:,:,1].view(s*b, h, w)
    #        py = position[:,0].reshape(-1)
    #        px = position[:,1].reshape(-1)
    #        x_polarity = x_polarity[range(s*b), py, px]
    #        y_polarity = torch.FloatTensor(
    #            y['%s_cursor'%space]['polarity']).view(-1).to(x_polarity.device)
    #        polarity_loss = binary_cross_entropy_with_logits(
    #            x_polarity, y_polarity, reduction='none').view(s,b)
    #        polarity_loss = polarity_loss * position_mask
    #        polarity_loss = torch.sum(polarity_loss) / (s*b)
    #        
    #        if log is not None:
    #            log.add_scalar(
    #                'train/%s_position_loss'%space, position_loss, clock[0])
    #            log.add_scalar(
    #                'train/%s_polarity_loss'%space, polarity_loss, clock[0])
    #            p = torch.softmax(x_position.view(s*b, -1), axis=-1)
    #            p = p.view(s*b, h, w)
    #            correct_p = p[range(s*b), position[:,0], position[:,1]]
    #            correct_p = correct_p.view(s,b) * position_mask
    #            pp = torch.sum(correct_p) / torch.sum(position_mask)
    #            log.add_scalar('train/%s_position_correct'%space, pp, clock[0])
    #            
    #            max_p = torch.argmax(p.view(s*b, -1), dim=-1)
    #            max_correct_p = (
    #                max_p.cpu() == (position[:,0] * w + position[:,1])).float()
    #            max_correct_p = max_correct_p.view(s,b) * position_mask.cpu()
    #            max_pp = (
    #                torch.sum(max_correct_p) / torch.sum(position_mask).cpu())
    #            log.add_scalar(
    #                'train/%s_max_position_correct'%space, max_pp, clock[0])

    #        position_losses.append(position_loss)
    #        polarity_losses.append(polarity_loss)
    #    
    #    workspace_position_loss, handspace_position_loss = position_losses
    #    workspace_polarity_loss, handspace_polarity_loss = polarity_losses
    #    
    #    loss = (
    #        mode_loss + class_loss + color_loss +
    #        workspace_position_loss + workspace_polarity_loss +
    #        handspace_position_loss + handspace_polarity_loss
    #    )
    #    
    #    return loss
    #    '''
    #
    #def tensor_to_actions(self, x, env, mode='sample'):
    #    return blocks_tensor_to_actions(x, mode=mode)
    #
    #'''
    #def tensor_to_actions(self, x, env, mode='sample'):
    #    #Convert model output tensors to gym actions
    #    s, b = x['mode'].shape[:2]
    #    assert s == 1, 'Expects a sequence length of 1, got %i'%s
    #    
    #    mode_action = categorical_or_max(x['mode'], mode=mode).cpu().numpy()
    #    class_action = categorical_or_max(x['class'], mode=mode).cpu().numpy()
    #    color_action = categorical_or_max(x['color'], mode=mode).cpu().numpy()
    #    
    #    workspace_yx = x['workspace'][:,:,0]
    #    workspace_y, workspace_x = categorical_or_max_2d(workspace_yx, mode)
    #    workspace_y = workspace_y.cpu().numpy()
    #    workspace_x = workspace_x.cpu().numpy()
    #    workspace_p = x['workspace'][:,:,1]
    #    workspace_p = bernoulli_or_max(workspace_p, mode).cpu().numpy()
    #    
    #    if 'handspace' in x:
    #        handspace_yx = x['handspace'][:,:,0]
    #        handspace_y, handspace_x = categorical_or_max_2d(handspace_yx, mode)
    #        handspace_y = handspace_y.cpu().numpy()
    #        handspace_x = handspace_x.cpu().numpy()
    #        handspace_p = x['handspace'][:,:,1]
    #        handspace_p = bernoulli_or_max(handspace_p, mode).cpu().numpy()
    #    
    #    actions = []
    #    for i in range(b):
    #        action = env.no_op_action()
    #        mode = mode_action[0,i]
    #        w_yx = numpy.array([workspace_y[0,i], workspace_x[0,i]])
    #        w_p = workspace_p[0,i,w_yx[0], w_yx[1]]
    #        if 'handspace' in x:
    #            h_yx = numpy.array([handspace_y[0,i], handspace_x[0,i]])
    #            h_p = handspace_p[0,i,h_yx[0], h_yx[1]]
    #        if mode == 0:
    #            action['workspace_cursor']['activate'] = 1
    #            action['workspace_cursor']['position'] = w_yx
    #            action['workspace_cursor']['polarity'] = w_p
    #            action['disassembly'] = 1

    #        elif mode >=1 and mode <=2:
    #            action['workspace_cursor']['activate'] = 1
    #            action['workspace_cursor']['position'] = w_yx
    #            action['workspace_cursor']['polarity'] = w_p
    #            action['handspace_cursor']['activate'] = 1
    #            action['handspace_cursor']['position'] = h_yx
    #            action['handspace_cursor']['polarity'] = h_p
    #            action['pick_and_place'] = mode

    #        elif mode >= 3 and mode <= 5:
    #            action['workspace_cursor']['activate'] = 1
    #            action['workspace_cursor']['position'] = w_yx
    #            action['workspace_cursor']['polarity'] = w_p
    #            action['rotate'] = mode -2

    #        elif mode >= 6 and mode <= 12:
    #            action['workspace_viewpoint'] = mode - 5

    #        elif mode >= 13 and mode <= 19:
    #            action['handspace_viewpoint'] = mode - 12

    #        elif mode == 20:
    #            action['phase_switch'] = 1

    #        elif mode == 21:
    #            action['phase_switch'] = 2

    #        elif mode == 22:
    #            action['insert_brick']['class_id'] = class_action[0,i]
    #            action['insert_brick']['color_id'] = color_action[0,i]

    #        actions.append(action)
    #    
    #    return actions
    #'''
    #
    #def visualize_episodes(self, epoch, episodes, visualization_directory):
    #    num_seqs = min(
    #        self.config.visualization_seqs, episodes.num_seqs())
    #    for seq_id in tqdm.tqdm(range(num_seqs)):
    #        seq_path = os.path.join(
    #            visualization_directory, 'seq_%06i'%seq_id)
    #        if not os.path.exists(seq_path):
    #            os.makedirs(seq_path)

    #        seq = episodes.get_seq(seq_id)
    #        seq_len = len_hierarchy(seq)
    #        workspace_frames = seq['observation']['workspace_color_render']
    #        handspace_frames = seq['observation']['handspace_color_render']
    #        for frame_id in range(seq_len):
    #            workspace_frame = workspace_frames[frame_id]
    #            handspace_frame = handspace_frames[frame_id]
    #            wh, ww = workspace_frame.shape[:2]
    #            hh, hw = handspace_frame.shape[:2]
    #            w = ww + hw
    #            joined_image = numpy.zeros((wh, w, 3), dtype=numpy.uint8)
    #            joined_image[:,:ww] = workspace_frame
    #            joined_image[wh-hh:,ww:] = handspace_frame

    #            def mode_string(action):
    #                result = []
    #                if action['disassembly']:
    #                    result.append('Disassembly')
    #                if action['insert_brick']['class_id'] != 0:
    #                    result.append('Insert Brick [%i] [%i]'%(
    #                        action['insert_brick']['class_id'],
    #                        action['insert_brick']['color_id'],
    #                    ))
    #                if action['pick_and_place'] == 1:
    #                    result.append('PickAndPlace [Cursor]')
    #                if action['pick_and_place'] == 2:
    #                    result.append('PickAndPlace [Origin]')
    #                if action['rotate']:
    #                    result.append('Rotate [%i]'%action['rotate'])
    #                if action['workspace_viewpoint']:
    #                    result.append('Workspace Viewpiont [%i]'%(
    #                        action['workspace_viewpoint']))
    #                if action['handspace_viewpoint']:
    #                    result.append('Handspace Viewpoint [%i]'%(
    #                        action['handspace_viewpoint']))
    #                if action['phase_switch']:
    #                    result.append(
    #                        'Phase Switch [%i]'%action['phase_switch'])
    #                return '\n'.join(result)
    #            
    #            def draw_workspace_dot(position, color, alpha=1.0):
    #                y, x = position
    #                joined_image[y*4:(y+1)*4, x*4:(x+1)*4] = (
    #                    color * alpha +
    #                    joined_image[y*4:(y+1)*4, x*4:(x+1)*4] * (1.-alpha)
    #                )
    #            
    #            def draw_handspace_dot(position, color):
    #                y, x = position
    #                yy = wh - hh
    #                joined_image[yy+y*4:yy+(y+1)*4, ww+x*4:ww+(x+1)*4] = (
    #                    color * alpha +
    #                    joined_image[yy+y*4:yy+(y+1)*4, ww+x*4:ww+(x+1)*4] *
    #                        (1. - alpha)
    #                )
    #            
    #            def draw_pick_and_place(action):
    #                if action['disassembly']:
    #                    pick = action['workspace_cursor']['position']
    #                    polarity = action['workspace_cursor']['polarity']
    #                    if polarity:
    #                        color = (0,0,255)
    #                    else:
    #                        color = (255,0,0)
    #                    draw_workspace_dot(pick, color)
    #                
    #                if action['pick_and_place']:
    #                    pick = action['handspace_cursor']['position']
    #                    draw_handspace_dot(pick, color)
    #                    place = action['workspace_cursor']['position']
    #                    draw_workspace_dot(place, color)
    #            
    #            lines = []
    #            frame_action = index_hierarchy(seq['action'], frame_id)
    #            lines.append(
    #                'Model:\n' + mode_string(frame_action) + '\n')
    #            lines.append('Reward: %f'%seq['reward'][frame_id])
    #            joined_image = write_text(joined_image, '\n'.join(lines))
    #            
    #            yx = frame_action['workspace_cursor']['position']
    #            p = frame_action['workspace_cursor']['polarity']
    #            if p:
    #                color = numpy.array((0,0,255))
    #            else:
    #                color = numpy.array((255,0,0))
    #            draw_workspace_dot(yx, color, 1.0)
    #            
    #            frame_path = os.path.join(
    #                seq_path,
    #                'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
    #            )
    #            save_image(joined_image, frame_path)
    
    #def eval_episodes(self, episodes, log, clock):
    #    num_seqs = min(
    #        self.config.visualization_seqs, episodes.num_seqs())
    #    ps = []
    #    rs = []
    #    f1s = []
    #    for seq_id in tqdm.tqdm(range(num_seqs)):
    #        seq = episodes.get_seq(seq_id)
    #        seq_len = len_hierarchy(seq)
    #        
    #        starting_config = (
    #            seq['observation']['initial_workspace_assembly'])
    #        starting_class = starting_config['class'][0]
    #        starting_color = starting_config['color'][0]
    #        starting_bricks = {}
    #        for class_id, color_id in zip(starting_class, starting_color):
    #            if class_id:
    #                brick = (class_id, color_id)
    #                starting_bricks.setdefault(brick, 0)
    #                starting_bricks[brick] += 1
    #        
    #        insert_class = seq['action']['insert_brick']['class_id']
    #        insert_color = seq['action']['insert_brick']['color_id']
    #        insert_bricks = {}
    #        for class_id, color_id in zip(insert_class, insert_color):
    #            if class_id:
    #                brick = (class_id, color_id)
    #                insert_bricks.setdefault(brick, 0)
    #                insert_bricks[brick] += 1
    #        
    #        all_brick_keys = (
    #            set(starting_bricks.keys()) | set(insert_bricks.keys()))
    #        tp = 0
    #        fp = 0
    #        fn = 0
    #        for brick in all_brick_keys:
    #            start_count = starting_bricks.get(brick, 0)
    #            insert_count = insert_bricks.get(brick, 0)
    #            tp += min(start_count, insert_count)
    #            difference = start_count - insert_count
    #            if difference < 0:
    #                fp += -difference
    #            elif difference > 0:
    #                fn += difference
    #        
    #        p, r = precision_recall(tp, fp, fn)
    #        f = f1(p, r)
    #        ps.append(p)
    #        rs.append(r)
    #        f1s.append(f)
    #    
    #    average_p = sum(ps)/len(ps)
    #    average_r = sum(rs)/len(rs)
    #    average_f1 = sum(f1s)/len(f1s)
    #    
    #    log.add_scalar('val/insert_precision', average_p, clock[0])
    #    log.add_scalar('val/insert_recall', average_r, clock[0])
    #    log.add_scalar('val/f1', average_f1, clock[0])
