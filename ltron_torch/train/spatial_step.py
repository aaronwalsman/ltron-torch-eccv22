import time
import math
import os

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy

import PIL.Image as Image

import tqdm

import splendor.masks as masks

from ltron.geometry.align import best_first_total_alignment
from ltron.bricks.brick_scene import BrickScene
from ltron.bricks.brick_shape import BrickShape

from ltron_torch.evaluation import spatial_metrics
from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron

from ltron_torch.envs.spatial_env import pose_estimation_env
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, gym_space_list_to_tensors)
'''
import ltron_torch.models.named_models as named_models
from ltron_torch.models.spatial import SE3Layer
from ltron_torch.models.mlp import Conv2dStack
from ltron_torch.models.vit_transformer import VITTransformerModel
'''
from ltron_torch.models.standard_models import single_step_model
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
        p_augment = 0.75,
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
        decoder_channels = 512,
        
        # test settings
        test_frequency = 1,
        test_steps_per_epoch = 1024,
        
        # checkpoint settings
        checkpoint_frequency = 1,
        
        # logging settings
        log_train = 0,
        log_test = 0):
    
    print('='*80)
    print('Setup')
    
    print('-'*80)
    print('Logging')
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    if random_floating_bricks:
        max_instances_per_scene += 20
    if random_floating_pairs:
        max_instances_per_scene += 40
    
    print('-'*80)
    print('Building the step model')
    step_model = single_step_model(
        model_backbone,
        num_classes=num_classes,
        decoder_channels=decoder_channels,
        pose_head=True,
        viewpoint_head=controlled_viewpoint,
    ).cuda()
    
    if step_checkpoint is not None:
        print('Loading step model checkpoint from:')
        print(step_checkpoint)
        step_model.load_state_dict(torch.load(step_checkpoint))
    
    print('-'*80)
    print('Building the optimizer')
    optimizer = torch.optim.Adam(
            list(step_model.parameters()),
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
        augment_dataset=augment_dataset,
        p_augment=p_augment,
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
                step_model,
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
                log_train)
        
        if (checkpoint_frequency is not None and
                epoch % checkpoint_frequency) == 0:
            checkpoint_directory = os.path.join(
                    './checkpoint', os.path.split(log.log_dir)[-1])
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            
            print('-'*80)
            step_model_path = os.path.join(
                    checkpoint_directory, 'step_model_%04i.pt'%epoch)
            print('Saving step_model to: %s'%step_model_path)
            torch.save(step_model.state_dict(), step_model_path)
            
            optimizer_path = os.path.join(
                    checkpoint_directory, 'optimizer_%04i.pt'%epoch)
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        if test_frequency and (epoch % test_frequency == 0):
            test_model(step_model, test_env, dataset_info, None)
        
        print('-'*80)
        print('Elapsed: %.04f'%(time.time() - epoch_start))


def train_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        mini_epochs,
        mini_epoch_sequences,
        mini_epoch_sequence_length,
        batch_size,
        step_model,
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
        log_train):
    
    print('-'*80)
    print('Train')
    
    # rollout ==================================================================
    
    print('- '*40)
    print('Rolling out episodes to generate data')
    
    seq_terminal = []
    seq_observations = []
    seq_rewards = []
    seq_viewpoint_actions = []
    seq_action_probs = []
    
    device = torch.device('cuda:0')
    
    step_model.eval()
    
    step_observations = train_env.reset()
    step_terminal = torch.ones(train_env.num_envs, dtype=torch.bool)
    step_rewards = numpy.zeros(train_env.num_envs)
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            
            # data storage and conversion --------------------------------------
            
            # store observation
            seq_terminal.append(step_terminal)
            seq_observations.append(step_observations)
            
            # convert gym observations to torch tensors
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    train_env.single_observation_space,
                    device)
            
            # forward pass -----------------------------------------------------
            head_features = step_model(step_tensors['color_render'])
            
            # act --------------------------------------------------------------
            actions = [{} for _ in range(train_env.num_envs)]
            if 'viewpoint' in head_features:
                viewpoint_logits = head_features['viewpoint']
                viewpoint_distribution = Categorical(logits=viewpoint_logits)
                viewpoint_actions = viewpoint_distribution.sample().cpu()
                for action, viewpoint_action in zip(actions, viewpoint_actions):
                    action['viewpoint'] = int(viewpoint_action)
                do_hide = numpy.array((viewpoint_actions == 0).float().cpu())
                seq_viewpoint_actions.append([
                    action['viewpoint'] for action in actions])
                step_action_probs = viewpoint_distribution.probs[
                    range(train_env.num_envs), viewpoint_actions].cpu().detach()
                seq_action_probs.append(step_action_probs)
            else:
                do_hide = numpy.ones(train_env.num_envs)
                
            unrolled_confidence_logits = head_features['confidence'].view(
                train_env.num_envs, -1)
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
            rotation_prediction = head_features['pose']['rotation'][
                range(train_env.num_envs), :, :, top_y, top_x]
            rotation_prediction_inv = rotation_prediction.permute((0, 2, 1))
            
            transform_labels = step_tensors['dense_pose_labels']
            rotation_labels = transform_labels[
                range(train_env.num_envs), top_y, top_x, :3, :3]
            
            rotation_offset = torch.matmul(
                rotation_labels, rotation_prediction)
            rotation_rewards = geometry.angle_surrogate(rotation_offset)
            
            # translation
            translation_prediction = head_features['pose']['translation'][
                range(train_env.num_envs), :, top_y, top_x]
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
            class_rewards = head_features['class_label'][
                range(train_env.num_envs), :, top_y, top_x]
            class_rewards = torch.softmax(class_rewards, dim=1)
            class_rewards = class_rewards[
                range(train_env.num_envs), class_label]
            
            step_rewards = (
                rotation_rewards * translation_rewards * class_rewards)
            step_rewards = step_rewards.cpu() #* do_hide
            #tmp_rewards = torch.softmax(head_features['viewpoint'], dim=-1)
            #tmp_rewards = tmp_rewards[:,1].cpu()
            #step_rewards = tmp_rewards
            
            seq_rewards.append(step_rewards)
            
            log.add_scalar(
                    'train_rollout/class_reward',
                    sum(class_rewards)/len(class_rewards),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/rotation_reward',
                    sum(rotation_rewards)/len(rotation_rewards),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/translation_reward',
                    sum(translation_rewards)/len(translation_rewards),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/total_reward',
                    sum(step_rewards)/len(step_rewards),
                    step_clock[0])
            
            #-------------------------------------------------------------------
            # step
            (step_observations,
             step_rewards,
             step_terminal,
             step_info) = train_env.step(actions)
            step_terminal = torch.BoolTensor(step_terminal)
            
            step_clock[0] += 1
    
    # data shaping =============================================================
    
    print('- '*40)
    print('Converting rollout data to tensors')
    
    # when joining these into one long list, make sure sequences are preserved
    seq_tensors = gym_space_list_to_tensors(
            seq_observations, train_env.single_observation_space)
    seq_terminal = torch.stack(seq_terminal, axis=1).reshape(-1)
    
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
    
    # supervision ==============================================================
    
    print('- '*40)
    print('Supervising rollout data')
    
    step_model.train()
    
    dataset_size = seq_tensors['color_render'].shape[0]
    tlast = 0
    for mini_epoch in range(1, mini_epochs+1):
        print('-   '*20)
        print('Training Mini Epoch: %i'%mini_epoch)
        
        # episode subsequences
        iterate = tqdm.tqdm(range(mini_epoch_sequences//batch_size))
        for seq_id in iterate:
            start_indices = torch.randint(dataset_size, (batch_size,))
            step_terminal = [True for _ in range(batch_size)]
            seq_loss = 0.
            
            # steps per episode subsequence
            for step in range(mini_epoch_sequence_length):
                step_indices = (start_indices + step) % dataset_size
                
                # update step terminal -----------------------------------------
                step_terminal = seq_terminal[step_indices]
                
                # move data to gpu ---------------------------------------------
                x_im = seq_tensors['color_render'][step_indices].cuda()
                x_seg = seq_tensors['segmentation_render'][step_indices].cuda()
                
                # step forward pass --------------------------------------------
                head_features = step_model(x_im)
                
                # compute loss -------------------------------------------------
                step_loss = 0.
                
                # policy gradient ----------------------------------------------
                if 'viewpoint' in head_features:
                    step_returns = seq_norm_returns[step_indices].cuda()
                    step_actions = seq_viewpoint_actions[step_indices].cuda()
                    action_dist = Categorical(logits=head_features['viewpoint'])
                    #logp = action_dist.log_prob(step_actions)
                    #policy_gradient_loss = torch.mean(
                    #    -logp * step_returns) * viewpoint_loss_weight
                    #step_loss = step_loss + policy_gradient_loss
                    #log.add_scalar(
                    #    'loss/policy_gradient',
                    #    policy_gradient_loss,
                    #    step_clock[0],
                    #)
                    old_probs = seq_action_probs[step_indices].cuda()
                    new_probs = action_dist.probs[
                        range(batch_size), step_actions]
                    prob_ratio = new_probs/old_probs
                    eps = 0.2
                    clipped_prob_ratio = torch.clamp(prob_ratio, 1-eps, 1+eps)
                    unclipped_advantage = prob_ratio * step_returns
                    clipped_advantage = clipped_prob_ratio * step_returns
                    ppo_loss = torch.min(unclipped_advantage, clipped_advantage)
                    ppo_loss = torch.mean(ppo_loss) * viewpoint_loss_weight
                    step_loss = step_loss + ppo_loss
                    log.add_scalar('loss/ppo', ppo_loss, step_clock[0])
                    
                    if entropy_loss_weight:
                        entropy_loss = - (
                            entropy_loss_weight *
                            torch.mean(action_dist.entropy())
                        )
                        step_loss = step_loss + entropy_loss
                        log.add_scalar(
                            'loss/entropy', entropy_loss, step_clock[0])
                
                # dense supervision --------------------------------------------
                class_label_logits = head_features['class_label']
                class_label_target = seq_tensors['dense_class_labels'][
                    step_indices, :, :, 0].cuda()
                class_label_loss = dense_class_label_loss(
                        class_label_logits,
                        class_label_target,
                        class_label_class_weight,
                )
                step_loss = (step_loss +
                        class_label_loss * class_label_loss_weight)
                log.add_scalar(
                    'loss/class_label', class_label_loss, step_clock[0])
                
                foreground = x_seg != 0
                foreground_total = torch.sum(foreground)
                if foreground_total:
                    
                    # rotation supervision -------------------------------------
                    rotation_prediction = head_features['pose']['rotation']
                    # this is both moving the matrix dimensions to the end
                    # and also transposing (inverting) the rotation matrix
                    #rotation_prediction = rotation_prediction.permute(
                    #    (0, 3, 4, 2, 1))
                    rotation_prediction = rotation_prediction.permute(
                        (0, 3, 4, 1, 2))
                    
                    transform_labels = seq_tensors['dense_pose_labels']
                    transform_labels = transform_labels[step_indices].cuda()
                    rotation_labels = transform_labels[:,:,:,:3,:3]
                    
                    '''
                    rotation_offset = torch.matmul(
                        rotation_labels, rotation_prediction)
                    angle_surrogate = geometry.angle_surrogate(rotation_offset)
                    
                    dense_rotation_loss = 1. - angle_surrogate
                    dense_rotation_loss = dense_rotation_loss * foreground
                    rotation_loss = (
                        torch.sum(dense_rotation_loss) / foreground_total)
                    '''
                    dense_rotation_loss = torch.nn.functional.mse_loss(
                        rotation_prediction,
                        rotation_labels,
                        reduction='none'
                    )
                    dense_rotation_loss = torch.sum(
                        dense_rotation_loss,
                        dim=(-2,-1),
                    ) * foreground
                    rotation_loss = (
                        torch.sum(dense_rotation_loss) / foreground_total)
                    
                    inverse_rotation_prediction = rotation_prediction.permute(
                        0, 1, 2, 4, 3)
                    rotation_offset = torch.matmul(
                        rotation_labels, inverse_rotation_prediction)
                    angle_surrogate = geometry.angle_surrogate(rotation_offset)
                    
                    step_loss = step_loss + rotation_loss * rotation_loss_weight
                    log.add_scalar(
                        'loss/rotation', rotation_loss, step_clock[0])
                    
                    # translation supervision ----------------------------------
                    translation_prediction = (
                        head_features['pose']['translation'])
                    translation_prediction = translation_prediction.permute(
                        (0, 2, 3, 1))
                    translation_prediction = (
                        translation_prediction * translation_scale_factor)
                    
                    translation_labels = (
                        transform_labels[:,:,:,:3,3] * translation_scale_factor)
                    offset = translation_prediction - translation_labels
                    dense_translation_loss = torch.sum(offset**2, dim=-1)
                    dense_translation_loss = dense_translation_loss * foreground
                    translation_loss = (
                        torch.sum(dense_translation_loss) / foreground_total)
                    
                    step_loss = (
                        step_loss + translation_loss * translation_loss_weight)
                    log.add_scalar(
                        'loss/translation', translation_loss, step_clock[0])
                    
                    distance_confidence = torch.exp(-dense_translation_loss*5)
                    
                    # score supervision ----------------------------------------
                    class_label_prediction = torch.argmax(
                            class_label_logits, dim=1)
                    class_correct = class_label_prediction == class_label_target
                    
                    score_target = (
                        class_correct *
                        angle_surrogate.detach() *
                        distance_confidence.detach()
                    )
                    
                    score_loss = dense_score_loss(
                        head_features['confidence'],
                        score_target,
                        foreground,
                    )
                    
                    step_loss = step_loss + score_loss * score_loss_weight
                    log.add_scalar('loss/score', score_loss, step_clock[0])
                    log.add_scalar('train_accuracy/dense_class_label',
                            float(torch.sum(score_target)) /
                            float(torch.numel(score_target)),
                            step_clock[0])
                    
                log.add_scalar('loss/total', step_loss, step_clock[0])
                
                seq_loss = seq_loss + step_loss
                
                step_clock[0] += 1
            
            seq_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test_checkpoint(
    step_checkpoint,
    dataset,
    num_processes=8,
    split='test',
    subset=None,
    image_resolution=(256,256),
    
    # model settings
    model_backbone = 'simple_fcn',
    decoder_channels = 512,
    
    # output settings
    output_path = None,
):
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    '''
    step_model = named_models.named_graph_step_model(
        step_model_name,
        backbone_name = step_model_backbone,
        decoder_channels = decoder_channels,
        num_classes = num_classes,
        input_resolution = image_resolution,
        pose_head = True,
        viewpoint_head = True,
    ).cuda()
    '''
    step_model = single_step_model(
        model_backbone,
        num_classes = num_classes,
        decoder_channels = decoder_channels,
        pose_head=True,
        viewpoint_head=False,
    ).cuda()
    step_model.load_state_dict(torch.load(step_checkpoint))
    
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
    
    test_model(step_model, env, dataset_info, output_path)

def test_model(
    step_model,
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
        for class_id, brick_shape in dataset_info['class_ids'].items()
    }
    
    class_boxes = {
        class_id : BrickShape(brick_shape).bbox
        for class_id, brick_shape in inverse_class_ids.items()
    }
    
    all_distances = []
    all_pair_distances = []
    all_snapped_distances = []
    all_snapped_pair_distances = []
    all_true_positives = 0
    all_false_positives = 0
    all_false_negatives = 0
    
    step_model.eval()
    
    with torch.no_grad():
        while not all_finished:
            step_tensors = gym_space_to_tensors(
                step_observations,
                env.single_observation_space,
                device,
            )
            
            head_features = step_model(step_tensors['color_render'])
            
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
                            'image_%i_%i.png'%(episode_id, frame_id),
                        )
                        image.save(image_path)
                        
                        confidence = torch.sigmoid(
                            head_features['confidence'][i,0])
                        confidence = (confidence * 255).byte()
                        confidence_image = Image.fromarray(
                            confidence.cpu().numpy())
                        confidence_path = os.path.join(
                            output_path,
                            'confidence_%i_%i.png'%(episode_id, frame_id),
                        )
                        confidence_image.save(confidence_path)
            
            # add to scenes ----------------------------------------------------
            class_prediction = torch.argmax(
                head_features['class_label'], dim=1).unsqueeze(1)
            confidence = torch.sigmoid(head_features['confidence'])
            segmentation_height, segmentation_width = confidence.shape[-2:]
            confidence = confidence * (class_prediction != 0)
            confidence = confidence.view(confidence.shape[0], -1)
            max_confidence_indices = torch.argmax(confidence, dim=-1)
            
            if output_path is not None:
                for i in range(env.num_envs):
                    if step_tensors['scene']['valid_scene_loaded'][i]:
                        episode_id = (
                            step_observations['dataset']['episode_id'][i])
                        frame_id = step_observations['episode_length'][i]
                        draw_conf = confidence[i].cpu()
                        draw_conf = draw_conf.view(-1,1).expand(-1,3).clone()
                        draw_conf[max_confidence_indices[i]] = (
                            torch.FloatTensor([1,0,0]))
                        draw_conf = draw_conf.view(64,64,3)
                        draw_conf = (draw_conf * 255).byte().numpy()
                        draw_conf_image = Image.fromarray(draw_conf)
                        draw_path = os.path.join(
                            output_path,
                            'selection_%i_%i.png'%(episode_id, frame_id),
                        )
                        draw_conf_image.save(draw_path)
            
            b = list(range(env.num_envs))
            max_class = class_prediction.view(
                env.num_envs, -1)[b, max_confidence_indices].cpu()
            max_rotation = head_features['pose']['rotation'].view(
                env.num_envs, 3, 3, -1)[b, :, :, max_confidence_indices].cpu()
            max_translation = head_features['pose']['translation'].view(
                env.num_envs, 3, -1)[b, :, max_confidence_indices].cpu()
            
            for i in range(env.num_envs):
                if step_tensors['scene']['valid_scene_loaded'][i]:
                    class_label = int(max_class[i])
                    if class_label != 0:
                        transform = numpy.eye(4)
                        transform[:3,:3] = numpy.array(max_rotation[i])
                        transform[:3,3] = numpy.array(max_translation[i])
                        camera_matrix = (
                            step_tensors['viewpoint']['camera_matrix'][i]).cpu()
                        camera_pose = numpy.linalg.inv(camera_matrix)
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
            
            if 'viewpoint' in head_features:
                viewpoint_actions = torch.argmax(
                    head_features['viewpoint'], dim=-1).cpu()
            
            for i in range(env.num_envs):
                actions[i]['visibility'] = select_actions[i]
                if 'viewpoint' in head_features:
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
                        class_id = dataset_info['class_ids'][
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
                        class_id = dataset_info['class_ids'][
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
    print('Average Snapped Pairwise Ditance:',
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
        #    class_id = dataset_info['class_ids'][class_name]
        #    transform = 
        if output_path is not None:
            mpd_path = os.path.join(
                output_path, 'reconstruction_%i_%i.mpd'%pid_sid)
            print(mpd_path)
            scene.export_ldraw(mpd_path)
            
            #best_first_total_alignment(scene)
            #mpd_path_snapped = os.path.join(
            #    output_path, 'reconstruction_snapped_%i_%i.mpd'%pid_sid)
            #scene.export_ldraw(mpd_path_snapped)
