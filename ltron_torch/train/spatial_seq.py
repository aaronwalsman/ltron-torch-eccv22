import time
import math
import os

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import numpy

import PIL.Image as Image

import tqdm

import splendor.masks as masks

from ltron.geometry.align import best_first_total_alignment
from ltron.geometry.relative_alignment import relative_alignment
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.brick_shape import BrickShape

from ltron_torch.evaluation import spatial_metrics
from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron
from ltron.gym.rollout_storage import (
    RolloutStorage,
    parallel_deepmap,
    stack_gym_data,
)

from ltron_torch.envs.spatial_env import pose_estimation_env
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, gym_space_list_to_tensors)
'''
import ltron_torch.models.named_models as named_models
from ltron_torch.models.spatial import SE3Layer
from ltron_torch.models.mlp import Conv2dStack
from ltron_torch.models.vit_transformer import VITTransformerModel
'''
import ltron_torch.models.standard_models as standard_models
from ltron_torch.models.seq_model import SeqCrossProductPoseModel
from ltron_torch.train.loss_old import (
        dense_class_label_loss, dense_score_loss, cross_product_loss)
import ltron_torch.geometry as geometry

def train_label_confidence(
    # load checkpoints
    step_checkpoint = None,
    edge_checkpoint = None,
    optimizer_checkpoint = None,
    
    # general settings
    num_epochs = 9999,
    mini_epochs_per_epoch = 1,
    mini_epoch_sequences = 2048,
    mini_epoch_sequence_length = 2,
    
    # dasaset settings
    dataset = 'random_stack',
    num_processes = 8,
    train_split = 'train',
    train_subset = None,
    test_split = 'test',
    test_subset = None,
    augment_dataset = None,
    image_resolution = (256,256),
    randomize_viewpoint = True,
    randomize_colors = True,
    random_floating_bricks = False,
    random_floating_pairs = False,
    random_bricks_per_scene = (10,20),
    random_bricks_subset = None,
    random_bricks_rotation_mode = None,
    controlled_viewpoint=False,
    
    # train settings
    train_steps_per_epoch = 4096,
    batch_size = 6,
    learning_rate = 3e-4,
    weight_decay = 1e-5,
    class_label_loss_weight = 0.8,
    class_label_background_weight = 0.05,
    score_loss_weight = 0.2,
    rotation_loss_weight = 1.0,
    translation_loss_weight = 1.0,
    translation_scale_factor = 0.01,
    viewpoint_loss_weight = 0.0,
    entropy_loss_weight = 0.0,
    max_instances_per_step = 8,
    
    # model settings
    model_backbone = 'simple_fcn',
    transformer_channels=256,
    decoder_channels=256,
    relative_poses=False,
    
    # test settings
    test_frequency = 1,
    test_steps_per_epoch = 1024,
    
    # checkpoint settings
    checkpoint_frequency = 1,
    
    # logging settings
    log_train = 0,
    log_test = 0
):
    
    print('='*80)
    print('Setup')
    
    print('-'*80)
    print('Logging')
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['shape_ids'].values()) + 1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    if random_floating_bricks:
        max_instances_per_scene += 20
    if random_floating_pairs:
        max_instances_per_scene += 40
    
    print('-'*80)
    print('Building the step model')
    seq_model = standard_models.seq_model(
        model_backbone,
        num_classes=num_classes,
        transformer_channels=transformer_channels,
        decoder_channels=decoder_channels,
        pose_head=True,
        viewpoint_head=controlled_viewpoint,
        translation_scale=translation_scale_factor,
    )
    
    if relative_poses:
        seq_model = SeqCrossProductPoseModel(
            seq_model, 'x', 'confidence', translation_scale_factor)
    
    seq_model = seq_model.cuda()
    
    if step_checkpoint is not None:
        print('Loading step model checkpoint from:')
        print(step_checkpoint)
        seq_model.load_state_dict(torch.load(step_checkpoint))
    
    print('-'*80)
    print('Building the optimizer')
    optimizer = torch.optim.Adam(
            list(seq_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay)
    if optimizer_checkpoint is not None:
        print('Loading optimizer checkpoint from:')
        print(optimizer_checkpoint)
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
    
    print('-'*80)
    print('Building the background class weight')
    class_label_class_weight = torch.ones(num_classes)
    class_label_class_weight[0] = class_label_background_weight
    class_label_class_weight = class_label_class_weight.cuda()
    
    print('-'*80)
    print('Building the train environment')
    if model_backbone == 'simple_fcn':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 4, image_resolution[1] // 4)
    elif model_backbone == 'vit':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 16, image_resolution[1] // 16)
    
    train_env = async_ltron(
        num_processes,
        pose_estimation_env,
        dataset=dataset,
        split=train_split,
        subset=train_subset,
        width=image_resolution[0],
        height=image_resolution[1],
        segmentation_width=segmentation_width,
        segmentation_height=segmentation_height,
        controlled_viewpoint=controlled_viewpoint,
    )
    
    if test_frequency:
        print('-'*80)
        print('Building the test environment')
        print('subset', test_subset)
        test_env = async_ltron(
            num_processes,
            pose_estimation_env,
            dataset=dataset,
            split=test_split,
            subset=test_subset,
            width=image_resolution[0],
            height=image_resolution[1],
            segmentation_width=segmentation_width,
            segmentation_height=segmentation_height,
            controlled_viewpoint=controlled_viewpoint,
            train=False,
        )
    
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_label_confidence_epoch(
            epoch,
            step_clock,
            log,
            math.ceil(train_steps_per_epoch / num_processes),
            mini_epochs_per_epoch,
            mini_epoch_sequences,
            mini_epoch_sequence_length,
            batch_size,
            seq_model,
            optimizer,
            train_env,
            class_label_loss_weight,
            class_label_class_weight,
            score_loss_weight,
            rotation_loss_weight,
            translation_loss_weight,
            translation_scale_factor,
            viewpoint_loss_weight,
            entropy_loss_weight,
            segmentation_width,
            segmentation_height,
            max_instances_per_step,
            max_instances_per_scene,
            dataset_info,
            log_train,
        )
        
        if (checkpoint_frequency is not None and
                epoch % checkpoint_frequency) == 0:
            checkpoint_directory = os.path.join(
                    './checkpoint', os.path.split(log.log_dir)[-1])
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            
            print('-'*80)
            seq_model_path = os.path.join(
                    checkpoint_directory, 'seq_model_%04i.pt'%epoch)
            print('Saving seq_model to: %s'%seq_model_path)
            torch.save(seq_model.state_dict(), seq_model_path)
            
            optimizer_path = os.path.join(
                    checkpoint_directory, 'optimizer_%04i.pt'%epoch)
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if test_frequency and (epoch % test_frequency == 0):
            test_model(seq_model, test_env, dataset_info, None)
        
        print('-'*80)
        print('Elapsed: %.04f'%(time.time() - epoch_start))

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
def seq_image_transform(x):
    seq = []
    for s in range(x.shape[0]):
        batch = []
        for b in range(x.shape[1]):
            batch.append(image_transform(x[s,b]))
        seq.append(torch.stack(batch))
    return torch.stack(seq)

def train_label_confidence_epoch(
    epoch,
    step_clock,
    log,
    steps,
    mini_epochs,
    mini_epoch_sequences,
    mini_epoch_sequence_length,
    batch_size,
    seq_model,
    optimizer,
    train_env,
    class_label_loss_weight,
    class_label_class_weight,
    score_loss_weight,
    rotation_loss_weight,
    translation_loss_weight,
    translation_scale_factor,
    viewpoint_loss_weight,
    entropy_loss_weight,
    segmentation_width,
    segmentation_height,
    max_instances_per_step,
    max_instances_per_scene,
    dataset_info,
    log_train,
):
    
    print('-'*80)
    print('Train')
    
    # rollout ==================================================================
    
    print('- '*40)
    print('Rolling out episodes to generate data')
    
    observation_storage = RolloutStorage(train_env.num_envs)
    action_reward_storage = RolloutStorage(train_env.num_envs)
    
    device = torch.device('cuda:0')
    
    seq_model.eval()
    
    step_observations = train_env.reset()
    step_terminal = numpy.ones(train_env.num_envs, dtype=numpy.bool)
    step_rewards = numpy.zeros(train_env.num_envs)
    with torch.no_grad():
        for step in tqdm.tqdm(range(steps)):
            
            # data storage and conversion --------------------------------------
            
            # store observation
            #rollout_storage.append_batch(
            #    terminal=terminal,
            #    observation=step_observations,
            #    reward=step_rewards
            #seq_terminal.append(step_terminal)
            #seq_observations.append(step_observations)
            
            action_reward_storage.start_new_sequences(step_terminal)
            observation_storage.start_new_sequences(step_terminal)
            observation_storage.append_batch(observation=step_observations)
            
            # convert gym observations to torch tensors
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    train_env.single_observation_space,
                    device)
            
            batch_observations, mask = observation_storage.get_current_seqs()
            
            # forward pass -----------------------------------------------------
            #head_features = seq_model(step_tensors['color_render'])
            x = batch_observations['observation']['color_render']
            x = seq_image_transform(x).cuda()
            mask = torch.BoolTensor(mask).cuda()
            head_features = seq_model(x, padding_mask=mask)
            final_seq_ids = [
                observation_storage.seq_len(seq) - 1
                for seq in observation_storage.batch_seq_ids
            ]
            
            local_features = {
                key:value for key, value in head_features.items()
                if ('global' not in key) and ('relative' not in key)
            }
            def index_fn(a):
                return a[final_seq_ids, list(range(train_env.num_envs))]
            final_local_features = parallel_deepmap(index_fn, local_features)
            
            # act --------------------------------------------------------------
            actions = [{} for _ in range(train_env.num_envs)]
            #if 'viewpoint' in head_features:
            if False:
                viewpoint_logits = head_features['viewpoint']
                viewpoint_distribution = Categorical(logits=viewpoint_logits)
                viewpoint_actions = viewpoint_distribution.sample().cpu()
                for action, viewpoint_action in zip(actions, viewpoint_actions):
                    action['viewpoint'] = int(viewpoint_action)
                do_hide = numpy.array((viewpoint_actions == 0).float().cpu())
                #seq_viewpoint_actions.append([
                #    action['viewpoint'] for action in actions])
                step_action_probs = viewpoint_distribution.probs[
                    range(train_env.num_envs), viewpoint_actions].cpu().detach()
                #seq_action_probs.append(step_action_probs)
            else:
                do_hide = numpy.ones(train_env.num_envs)
            
            unrolled_confidence_logits = (
                final_local_features['confidence'].view(train_env.num_envs, -1))
            unrolled_confidence = torch.sigmoid(unrolled_confidence_logits)
            unrolled_segmentation = step_tensors['segmentation_render'].view(
                train_env.num_envs, -1)
            unrolled_confidence = (
                unrolled_confidence * (unrolled_segmentation != 0))
            select_actions = numpy.zeros(
                (train_env.num_envs, segmentation_height, segmentation_width),
                dtype=numpy.bool)
            selected_yx = []
            for i in range(max_instances_per_step):
                max_confidence_indices = torch.argmax(
                    unrolled_confidence, dim=-1)
                y = (max_confidence_indices // segmentation_height).cpu()
                x = (max_confidence_indices % segmentation_height).cpu()
                select_actions[range(train_env.num_envs), y, x] = True
                
                selected_yx.append((y,x))
                
                # turn off the selected brick instance
                instance_ids = unrolled_segmentation[
                    range(train_env.num_envs), max_confidence_indices]
                unrolled_confidence = (
                    unrolled_confidence *
                    (unrolled_segmentation != instance_ids.unsqueeze(1)))
            
            select_actions = select_actions * do_hide.reshape((-1, 1, 1))
            for i in range(train_env.num_envs):
                actions[i]['visibility'] = select_actions[i]
            
            # compute reward ---------------------------------------------------
            top_y, top_x = selected_yx[0]
            
            # rotation
            rotation_prediction = final_local_features['pose']['rotation'][
                range(train_env.num_envs), :, :, top_y, top_x]
            global_rotation_prediction = (
                head_features['global_pose']['rotation'])
            rotation_prediction = torch.matmul(
                global_rotation_prediction,
                rotation_prediction,
            )
            rotation_prediction_inv = rotation_prediction.permute((0, 2, 1))
            
            transform_labels = step_tensors['dense_pose_labels']
            rotation_labels = transform_labels[
                range(train_env.num_envs), top_y, top_x, :3, :3]
            
            rotation_offset = torch.matmul(
                rotation_labels, rotation_prediction)
            rotation_rewards = geometry.angle_surrogate(rotation_offset)
            
            # translation
            translation_prediction = (
                final_local_features['pose']['translation'][
                range(train_env.num_envs), :, top_y, top_x]
            )
            global_translation_prediction = (
                head_features['global_pose']['translation'])
            translation_prediction = torch.matmul(
                global_rotation_prediction,
                translation_prediction.unsqueeze(-1),
            ).squeeze(-1) + global_translation_prediction
            translation_labels = transform_labels[
                range(train_env.num_envs), top_y, top_x, :3, 3]
            translation_offset = (
                (translation_labels - translation_prediction) *
                translation_scale_factor)
            squared_distance = torch.sum(translation_offset**2, dim=-1)
            translation_rewards = torch.exp(-squared_distance*5)
            
            # class
            class_label = step_tensors['dense_class_labels'][
                range(train_env.num_envs), top_y, top_x, 0]
            class_rewards = final_local_features['class_label'][
                range(train_env.num_envs), :, top_y, top_x]
            class_rewards = torch.softmax(class_rewards, dim=1)
            class_rewards = class_rewards[
                range(train_env.num_envs), class_label]
            
            tmp_rewards = (
                rotation_rewards * translation_rewards * class_rewards)
            tmp_rewards = tmp_rewards.cpu().numpy() #* do_hide
            
            log.add_scalar(
                    'train_rollout/class_reward',
                    float(sum(class_rewards)/len(class_rewards)),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/rotation_reward',
                    float(sum(rotation_rewards)/len(rotation_rewards)),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/translation_reward',
                    float(sum(translation_rewards)/len(translation_rewards)),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/total_reward',
                    float(sum(tmp_rewards)/len(tmp_rewards)),
                    step_clock[0])
            
            # step -------------------------------------------------------------
            (step_observations,
             step_rewards,
             step_terminal,
             step_info) = train_env.step(actions)
            step_terminal = torch.BoolTensor(step_terminal)
            
            # hack for now...
            step_rewards = tmp_rewards
            
            # storage ----------------------------------------------------------
            action_reward_storage.append_batch(
                action=stack_gym_data(*actions),
                reward=step_rewards,
            )
            
            step_clock[0] += 1
    
    # data shaping =============================================================
    
    print('- '*40)
    print('Converting rollout data to tensors')
    
    '''
    # when joining these into one long list, make sure sequences are preserved
    seq_tensors = gym_space_list_to_tensors(
            seq_observations, train_env.single_observation_space)
    seq_terminal = torch.stack(seq_terminal, axis=1).reshape(-1)
    '''
    
    '''
    # compute returns
    seq_returns = []
    ret = 0.
    gamma = 0.9
    seq_rewards = torch.stack(seq_rewards, axis=1).reshape(-1)
    for reward, terminal in zip(reversed(seq_rewards), reversed(seq_terminal)):
        ret += float(reward)
        seq_returns.append(ret)
        ret *= gamma
        if terminal:
            ret = 0.
    
    seq_returns = torch.FloatTensor(list(reversed(seq_returns)))
    
    seq_norm_returns = (
            (seq_returns - torch.mean(seq_returns)) /
            (seq_returns.std() + 0.001))
    
    if len(seq_viewpoint_actions):
        seq_viewpoint_actions = torch.LongTensor([seq_viewpoint_actions[i][j]
                for j in range(train_env.num_envs)
                for i in range(steps)])
        seq_action_probs = torch.stack(seq_action_probs, axis=1).reshape(-1)
    '''
    
    # supervision ==============================================================
    
    print('- '*40)
    print('Supervising rollout data')
    
    seq_model.train()
    
    rollout_storage = observation_storage | action_reward_storage
    
    assert min(
        rollout_storage.seq_len(i) for i in range(rollout_storage.num_seqs())
    ) >= 1
    
    tlast = 0
    for mini_epoch in range(1, mini_epochs+1):
        
        print('-   '*20)
        print('Training Mini Epoch: %i'%mini_epoch)
        
        iterate = tqdm.tqdm(rollout_storage.batch_sequence_iterator(
            batch_size,
            shuffle=True,
        ))
        for batch_data, batch_mask in iterate:
            
            # forward pass -----------------------------------------------------
            x = batch_data['observation']['color_render']
            x = seq_image_transform(x).cuda()
            batch_padding_mask = torch.BoolTensor(batch_mask).cuda()
            seq_len = x.shape[0]
            seq_mask = torch.triu(
                torch.ones(seq_len, seq_len),
                diagonal=1,
            ).cuda()
            head_features = seq_model(
                x,
                seq_mask=seq_mask,
                padding_mask=batch_padding_mask,
            )
            
            # loss -------------------------------------------------------------
            seq_loss = 0.
            
            # policy gradient --------------------------------------------------
            pass
            
            # dense supervision ------------------------------------------------
            class_label_logits = head_features['class_label']
            s, b, c, h, w = class_label_logits.shape
            class_label_logits = class_label_logits.view(s*b, c, h, w)
            
            class_label_target = batch_data['observation']['dense_class_labels']
            class_label_target = torch.LongTensor(
                class_label_target).cuda()[...,0]
            class_label_target = class_label_target.view(s*b, h, w)
            
            inv_padding_mask = ~batch_padding_mask.transpose(0, 1).reshape(s*b)
            class_label_loss = dense_class_label_loss(
                    class_label_logits[inv_padding_mask],
                    class_label_target[inv_padding_mask],
                    class_label_class_weight,
            )
            
            seq_loss = (seq_loss +
                    class_label_loss * class_label_loss_weight)
            log.add_scalar(
                'loss/class_label', float(class_label_loss), step_clock[0])
            
            foreground = batch_data['observation']['segmentation_render'] != 0
            foreground = torch.BoolTensor(foreground).cuda()
            
            foreground_total = torch.sum(foreground)
            if foreground_total:
                
                # rotation supervision -----------------------------------------
                rotation_prediction = head_features['pose']['rotation']
                if 'global_pose' in head_features:
                    global_rotation = head_features['global_pose']['rotation']
                    b = global_rotation.shape[0]
                    global_rotation = global_rotation.view(1, b, 1, 1, 3, 3)
                    
                    '''
                    # this is both moving the matrix dimensions to the end
                    # and also transposing (inverting) the rotation matrix
                    rotation_prediction = rotation_prediction.permute(
                        #(0, 3, 4, 2, 1))
                        (0, 1, 4, 5, 3, 2))
                    '''
                    rotation_prediction = rotation_prediction.permute(
                        (0, 1, 4, 5, 2, 3))
                    rotation_prediction = torch.matmul(
                        global_rotation,
                        rotation_prediction,
                    )
                    
                    # inverse
                    inverse_rotation_prediction = rotation_prediction.permute(
                        (0, 1 ,2, 3, 5, 4))
                
                else:
                    inverse_rotation_prediction = rotation_prediction.permute(
                        (0, 1, 4, 5, 3, 2))
                
                #transform_labels = seq_tensors['dense_pose_labels']
                transform_labels = torch.FloatTensor(
                    batch_data['observation']['dense_pose_labels']).cuda()
                rotation_labels = transform_labels[:,:,:,:,:3,:3]
                
                rotation_offset = torch.matmul(
                    rotation_labels, inverse_rotation_prediction)
                angle_surrogate = geometry.angle_surrogate(rotation_offset)
                
                dense_rotation_loss = 1. - angle_surrogate
                dense_rotation_loss = dense_rotation_loss * foreground
                rotation_loss = (
                    torch.sum(dense_rotation_loss) / foreground_total)
                
                seq_loss = seq_loss + rotation_loss * rotation_loss_weight
                log.add_scalar(
                    'loss/rotation', float(rotation_loss), step_clock[0])
                
                # relative rotation supervision --------------------------------
                if 'relative_indices' in head_features:
                    #selected_indices = head_features[
                    #    'relative_indices'].view(-1)
                    rollout_selections = torch.LongTensor(
                        batch_data['action']['visibility'])
                    s, b, h, w = rollout_selections.shape
                    rollout_selections = rollout_selections.view(s*b, h*w)
                    #valid_frames = torch.sum(rollout_selections, dim=-1)
                    selected_indices = torch.argmax(rollout_selections, dim=-1)
                    
                    selected_pose_labels = torch.FloatTensor(
                        batch_data['observation']['dense_pose_labels']).cuda()
                    s, b, h, w, _, _ = selected_pose_labels.shape
                    selected_pose_labels = selected_pose_labels.view(
                        s*b, h*w, 4, 4)
                    selected_pose_labels = selected_pose_labels[
                        range(s*b), selected_indices]
                    selected_pose_labels = selected_pose_labels.view(s, b, 4, 4)
                    square_pose_labels = selected_pose_labels.view(
                        s, 1, b, 4, 4)
                    inverse_pose_labels = selected_pose_labels.view(s*b, 4, 4)
                    inverse_pose_labels[batch_mask.T.reshape(-1)] = (
                        torch.eye(4).cuda())
                    inverse_pose_labels = torch.inverse(inverse_pose_labels)
                    inverse_pose_labels = inverse_pose_labels.reshape(
                        1, s, b, 4, 4)
                    offsets = torch.matmul(
                        inverse_pose_labels, square_pose_labels)
                    
                    relative_rotation_loss = torch.nn.functional.mse_loss(
                        offsets[:,:,:,:3,:3],
                        head_features['relative_poses']['rotation'],
                        reduction='none',
                    )
                    relative_rotation_loss = torch.mean(
                        relative_rotation_loss,
                        dim=(-2,-1),
                    )
                    masked_instances = torch.BoolTensor(batch_mask.T)
                    square_mask = ~(
                        masked_instances.unsqueeze(0) |
                        masked_instances.unsqueeze(1)
                    ).cuda()
                    total_instances = torch.sum(square_mask)
                    relative_rotation_loss = torch.sum(
                        relative_rotation_loss *
                        square_mask
                    ) / total_instances
                    seq_loss = seq_loss + (
                        relative_rotation_loss * rotation_loss_weight)
                    log.add_scalar(
                        'loss/relative-rotation',
                        float(relative_rotation_loss),
                        step_clock[0],
                    )
                    
                    # translation ---------
                    relative_translation = (
                        head_features['relative_poses']['translation'])
                    relative_translation = (
                        relative_translation * translation_scale_factor)
                    relative_translation_loss = torch.nn.functional.mse_loss(
                        offsets[:,:,:,:3,3] * translation_scale_factor,
                        relative_translation,
                        reduction='none',
                    )
                    relative_translation_loss = torch.mean(
                        relative_translation_loss, dim=-1)
                    relative_translation_loss = torch.sum(
                        relative_translation_loss *
                        square_mask
                    ) / total_instances
                    seq_loss = seq_loss + (
                        relative_translation_loss * translation_loss_weight)
                    log.add_scalar(
                        'loss/relative-translation',
                        float(relative_translation_loss),
                        step_clock[0],
                    )
                
                # translation supervision --------------------------------------
                translation_prediction = (
                    head_features['pose']['translation'])
                global_translation = (
                    head_features['global_pose']['translation']
                ).view(1, b, 1, 1, 3, 1)
                translation_prediction = translation_prediction.permute(
                    #(0, 2, 3, 1))
                    (0, 1, 3, 4, 2))
                translation_prediction = translation_prediction.unsqueeze(-1)
                translation_prediction = torch.matmul(
                    global_rotation,
                    translation_prediction,
                ) + global_translation
                s, b, h, w, _, _ = translation_prediction.shape
                translation_prediction = translation_prediction.view(
                    s, b, h, w, 3)
                
                # This is done here because the output of the SE3 layer scales
                # the translation up, this scales it back down so it can be
                # compared against the target (which is also scaled down) at
                # a reasonable scale.
                translation_prediction = (
                    translation_prediction * translation_scale_factor)
                
                translation_labels = (
                    transform_labels[:,:,:,:,:3,3] * translation_scale_factor)
                offset = translation_prediction - translation_labels
                dense_translation_loss = torch.sum(offset**2, dim=-1)
                dense_translation_loss = dense_translation_loss * foreground
                translation_loss = (
                    torch.sum(dense_translation_loss) / foreground_total)
                
                seq_loss = (
                    seq_loss + translation_loss * translation_loss_weight)
                log.add_scalar(
                    'loss/translation', float(translation_loss), step_clock[0])
                
                distance_confidence = torch.exp(-dense_translation_loss*5)
                
                # score supervision --------------------------------------------
                class_label_prediction = torch.argmax(
                        class_label_logits, dim=1)
                class_correct = class_label_prediction == class_label_target
                
                score_target = (
                    class_correct *
                    angle_surrogate.view(s*b, h, w).detach() *
                    distance_confidence.view(s*b, h, w).detach()
                )
                
                score_loss = dense_score_loss(
                    head_features['confidence'].view(s*b, 1, h, w),
                    score_target,
                    foreground.view(s*b, h, w),
                )
                
                seq_loss = seq_loss + score_loss * score_loss_weight
                log.add_scalar('loss/score', score_loss, step_clock[0])
                log.add_scalar('train_accuracy/dense_class_label',
                        float(torch.sum(score_target)) /
                        float(torch.numel(score_target)),
                        step_clock[0])
                
            log.add_scalar('loss/total', seq_loss, step_clock[0])
            
            step_clock[0] += 1
            
            seq_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            
def test_checkpoint(
    seq_checkpoint,
    dataset,
    num_processes=8,
    split='test',
    subset=None,
    image_resolution=(256,256),
    translation_scale_factor=0.01,
    
    # model settings
    model_backbone = 'simple_fcn',
    transformer_channels = 256,
    decoder_channels = 256,
    relative_poses=False,
    
    # output settings
    output_path = None,
):
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['shape_ids'].values()) + 1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    seq_model = standard_models.seq_model(
        model_backbone,
        num_classes = num_classes,
        transformer_channels = transformer_channels,
        decoder_channels = decoder_channels,
        pose_head=True,
        viewpoint_head=False,
        translation_scale=translation_scale_factor,
    )
    if relative_poses:
        seq_model = SeqCrossProductPoseModel(
            seq_model, 'x', 'confidence', translation_scale_factor)
    
    seq_model = seq_model.cuda()
    
    seq_model.load_state_dict(torch.load(seq_checkpoint))
    
    if model_backbone == 'simple_fcn':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 4, image_resolution[1] // 4)
    elif model_backbone == 'vit':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 16, image_resolution[1] // 16)
    
    env = async_ltron(
        num_processes,
        pose_estimation_env,
        dataset=dataset,
        split=split,
        subset=subset,
        width=image_resolution[0],
        height=image_resolution[1],
        segmentation_width=segmentation_width,
        segmentation_height=segmentation_height,
        controlled_viewpoint=False,
        train=False,
    )
    
    test_model(seq_model, env, dataset_info, output_path)

def test_model(
    seq_model,
    env,
    dataset_info,
    output_path,
):
    
    device = torch.device('cuda:0')
    
    step_observations = env.reset()
    
    step_terminal = numpy.ones(env.num_envs, dtype=numpy.bool)
    all_finished = False
    scene_index = [0 for _ in range(env.num_envs)]
    reconstructed_scenes = {}
    active_scenes = [
        BrickScene(renderable=False, track_snaps=True)
        for _ in range(env.num_envs)
    ]
    target_scenes = [
        BrickScene(renderable=False, track_snaps=True)
        for _ in range(env.num_envs)
    ]
    
    inverse_class_ids = {
        brick_shape : class_id
        for class_id, brick_shape in dataset_info['shape_ids'].items()
    }
    
    class_boxes = {
        class_id : BrickShape(brick_shape).bbox
        for class_id, brick_shape in inverse_class_ids.items()
    }
    
    observation_storage = RolloutStorage(env.num_envs)
    action_reward_storage = RolloutStorage(env.num_envs)
    
    all_distances = []
    all_pair_distances = []
    all_snapped_distances = []
    all_snapped_pair_distances = []
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    
    seq_model.eval()
    
    with torch.no_grad():
        while not all_finished:
            step_tensors = gym_space_to_tensors(
                step_observations,
                env.single_observation_space,
                device,
            )
            
            action_reward_storage.start_new_sequences(step_terminal)
            observation_storage.start_new_sequences(step_terminal)
            observation_storage.append_batch(observation=step_observations)
            
            if output_path is not None:
                for i in range(env.num_envs):
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        episode_id = (
                            step_observations['dataset']['episode_id'][i])
                        frame_id = step_observations['episode_length'][i]
                        image = Image.fromarray(
                            step_observations['color_render'][i])
                        image_path = os.path.join(
                            output_path,
                            'image_%i_%i.png'%(episode_id, frame_id)
                        )
                        image.save(image_path)
            
            #head_features = seq_model(step_tensors['color_render'])
            batch_observations, mask = observation_storage.get_current_seqs()
            x = batch_observations['observation']['color_render']
            x = seq_image_transform(x).cuda()
            mask = torch.BoolTensor(mask).cuda()
            head_features = seq_model(x, padding_mask=mask)
            final_seq_ids = [
                observation_storage.seq_len(seq) - 1
                for seq in observation_storage.batch_seq_ids
            ]
            local_features = {
                key:value for key, value in head_features.items()
                if 'global' not in key and 'relative' not in key
            }
            def index_fn(a):
                return a[final_seq_ids, list(range(env.num_envs))]
            final_local_features = parallel_deepmap(index_fn, local_features)
            
            # add to scenes ----------------------------------------------------
            class_prediction = torch.argmax(
                final_local_features['class_label'], dim=1).unsqueeze(1)
            confidence = torch.sigmoid(final_local_features['confidence'])
            segmentation_height, segmentation_width = confidence.shape[-2:]
            confidence = confidence * (class_prediction != 0)
            confidence = confidence.view(confidence.shape[0], -1)
            max_confidence_indices = torch.argmax(confidence, dim=-1)
            
            b = list(range(env.num_envs))
            max_class = class_prediction.view(
                env.num_envs, -1)[b, max_confidence_indices].cpu()
            max_rotation = final_local_features['pose']['rotation'].view(
                env.num_envs, 3, 3, -1)[b, :, :, max_confidence_indices]
            global_rotation = head_features['global_pose']['rotation']
            
            max_rotation = torch.matmul(global_rotation, max_rotation).cpu()
            max_translation = final_local_features['pose']['translation'].view(
                env.num_envs, 3, -1)[b, :, max_confidence_indices]
            global_translation = head_features['global_pose']['translation']
            max_translation = (torch.matmul(
                global_rotation,
                max_translation.unsqueeze(-1),
            ).squeeze(-1) + global_translation).cpu()
            
            for i in range(env.num_envs):
                if step_tensors['scene']['valid_scene_loaded'][i]:
                    class_label = int(max_class[i])
                    if class_label != 0:
                        transform = numpy.eye(4)
                        transform[:3,:3] = numpy.array(max_rotation[i])
                        transform[:3,3] = numpy.array(max_translation[i])
                        view_matrix = (
                            step_tensors['viewpoint']['view_matrix'][i]).cpu()
                        camera_pose = numpy.linalg.inv(view_matrix)
                        active_scenes[i].add_instance(
                            brick_shape = inverse_class_ids[class_label],
                            brick_color = 4,
                            transform = transform,
                        )
            
            # act --------------------------------------------------------------
            actions = [{} for _ in range(env.num_envs)]
            select_actions = numpy.zeros(
                (env.num_envs, segmentation_height, segmentation_width),
                dtype=numpy.bool)
            y = (max_confidence_indices // segmentation_height).cpu()
            x = (max_confidence_indices % segmentation_height).cpu()
            select_actions[range(env.num_envs), y, x] = True
            
            if 'viewpoint' in final_local_features:
                viewpoint_actions = torch.argmax(
                    final_local_features['viewpoint'], dim=-1).cpu()
            
            for i in range(env.num_envs):
                actions[i]['visibility'] = select_actions[i]
                if 'viewpoint' in final_local_features:
                    actions[i]['viewpoint'] = int(viewpoint_actions[i])
            
            # step -------------------------------------------------------------
            (step_observations,
             step_rewards,
             step_terminal,
             step_info) = env.step(actions)
            step_terminal = torch.BoolTensor(step_terminal)
            
            # store terminal scenes --------------------------------------------
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    
                    # build transforms
                    s, s, b = (
                        head_features['relative_poses']['rotation'].shape[:3])
                    relative_rotations = (
                        head_features['relative_poses']['rotation'][:,:,i])
                    relative_translations = (
                        head_features['relative_poses']['translation'][:,:,i])
                    transforms = numpy.zeros((s, s, 4, 4))
                    transforms[:,:,3,3] = 1.
                    transforms[:,:,:3,:3] = relative_rotations.cpu().numpy()
                    transforms[:,:,:3,3] = relative_translations.cpu().numpy()
                    start_transforms = numpy.zeros((s, 4, 4))
                    start_transforms[:] = numpy.eye(4)
                    configuration = relative_alignment(
                        start_transforms, transforms, 10)
                    
                    s, b, n, h, w = head_features['class_label'].shape
                    predicted_class_ids = torch.argmax(
                        head_features['class_label'], dim=2)[:,i]
                    predicted_class_ids = predicted_class_ids.view(s, h*w)
                    relative_indices = head_features['relative_indices'][:,i]
                    selected_class_ids = predicted_class_ids[
                        range(s), relative_indices]
                    new_scene = BrickScene()
                    for transform, class_label in zip(
                        configuration, selected_class_ids):
                        class_label = int(class_label)
                        if class_label == 0:
                            continue
                        new_scene.add_instance(
                            brick_shape = inverse_class_ids[class_label],
                            brick_color = 4,
                            transform = transform,
                        )
                    new_scene.export_ldraw('./tmp.mpd')
                    
                    for j in range(s):
                        new_scene = BrickScene()
                        for transform, class_label in zip(
                            transforms[:,j], selected_class_ids
                        ):
                            class_label = int(class_label)
                            if class_label == 0:
                                continue
                            new_scene.add_instance(
                                brick_shape = inverse_class_ids[class_label],
                                brick_color = 4,
                                transform = transform,
                            )
                        new_scene.export_ldraw('./tmp_%i.mpd'%j)
                    
                    import pdb
                    pdb.set_trace()
                    
                    # compare reconstructed scene and target scene -------------
                    target_class_ids = (
                        step_tensors['class_labels'][i].cpu())
                    target_poses = step_tensors['pose_labels'][i].cpu()
                    target_instances = [
                        (int(target_id), numpy.array(target_pose))
                        for target_id, target_pose
                        in zip(target_class_ids, target_poses)
                        if target_id != 0
                    ]
                    scene = active_scenes[i]
                    predicted_instances = []
                    for instance_id, instance in scene.instances.items():
                        class_id = dataset_info['shape_ids'][
                            str(instance.brick_shape)]
                        predicted_instances.append(
                            (class_id, instance.transform))
                    
                    if len(target_instances) or len(predicted_instances):
                        distances, pair_distances, tp, fp, fn = spatial_metrics(
                            target_instances,
                            predicted_instances,
                            class_boxes,
                        )
                        all_distances.extend(distances)
                        all_pair_distances.extend(pair_distances)
                        all_true_positives += tp
                        all_false_positives += fp
                        all_false_negatives += fn
                    
                    # snap all the parts togther
                    #best_first_total_alignment(scene)
                    
                    # now get a new predicted instance list
                    predicted_instances = []
                    for instance_id, instance in scene.instances.items():
                        class_id = dataset_info['shape_ids'][
                            str(instance.brick_shape)]
                        predicted_instances.append(
                            (class_id, instance.transform))
                    
                    if len(target_instances) or len(predicted_instances):
                        distances, pair_distances, tp, fp, fn = spatial_metrics(
                            target_instances,
                            predicted_instances,
                            class_boxes,
                        )
                        all_snapped_distances.extend(distances)
                        all_snapped_pair_distances.extend(pair_distances)
                    
                    # make new empty scene -------------------------------------
                    if len(active_scenes[i].instances):
                        reconstructed_scenes[i, scene_index[i]] = (
                            active_scenes[i])
                        active_scenes[i] = BrickScene(
                            renderable=False, track_snaps=True)
                        scene_index[i] += 1
                    
                    # progress -------------------------------------------------
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        print('Loading scene %i for process %i'%(
                            scene_index[i], i))
            
            # all finished? ----------------------------------------------------
            all_finished = torch.all(
                step_tensors['scene']['valid_scene_loaded'] == 0)
    
    print('Average Distance:', sum(all_distances)/len(all_distances))
    print('Average Pairwise Distance:',
        sum(all_pair_distances)/len(all_pair_distances))
    print('Average Snapped Distance:',
        sum(all_snapped_distances)/len(all_snapped_distances))
    print('Average Snapped Pairwise Distance:',
        sum(all_snapped_pair_distances)/len(all_snapped_pair_distances))
    print('Precision:',
        (all_true_positives / (all_true_positives + all_false_positives)))
    print('Recall:',
        (all_true_positives / (all_true_positives + all_false_negatives)))
    
    
    # compute scene statistics =================================================
    for pid_sid, scene in tqdm.tqdm(reconstructed_scenes.items()):
        #instances = []
        #for instance_id, instance in scene.items:
        #    class_name = str(instance.brick_shape)
        #    class_id = dataset_info['shape_ids'][class_name]
        #    transform = 
        if output_path is not None:
            mpd_path = os.path.join(
                output_path, 'reconstruction_%i_%i.mpd'%pid_sid)
            scene.export_ldraw(mpd_path)
            
            #best_first_total_alignment(scene)
            #mpd_path_snapped = os.path.join(
            #    output_path, 'reconstruction_snapped_%i_%i.mpd'%pid_sid)
            #scene.export_ldraw(mpd_path_snapped)
