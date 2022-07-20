import random

import numpy

from ltron.hierarchy import index_hierarchy

def make_reassembly_labels(seq):
    observation_seq = seq['observation']
    reassembly_obs = observation_seq['reassembly']
    reassembling_steps = reassembly_obs['reassembling']
    disassembling_steps = (1 - reassembling_steps).astype(numpy.bool)
    workspace_configurations = reassembly_obs['workspace_configuration']
    target_configuration = index_hierarchy(
        reassembly_obs['target_configuration'], 0)
    
    disassembly_obs = observation_seq['disassembly']
    #disassembly_order = []
    
    labels = []
    
    # disassembly ==============================================================
    
    first_reassembly_step = disassembling_steps.shape[0]
    for i in range(disassembling_steps.shape[0]):
        
        # if we are no longer disassembling, stop making disassembly labels
        if not disassembling_steps[i]:
            first_reassembly_step = i
            break
        
        # if a brick was successfully disassembled, store it for later
        # in order to supervise reassembly
        #if disassembly_obs['success'][i]:
        #    disassembly_order.append(disassembly_obs['instance_id'][i])
        
        # if there are no bricks remaining, supervise switching to reassembly
        if not numpy.any(workspace_configurations['class'][i]):
            label = handspace_reassembly_template_action()
            label['reassembly']['start'] = 1
            labels.append(label)
            continue
        
        # if nothing was successfully disassembled this step
        # supervise a random visible removable brick
        #if not disassembly_obs['success'][i]:
        if True:
            segmentations = observation_seq['workspace_segmentation_render'][i]
            pos_snaps = observation_seq['workspace_pos_snap_render'][i]
            neg_snaps = observation_seq['workspace_neg_snap_render'][i]
            visible_removable_snaps = get_visible_removable_snaps(
                segmentations, pos_snaps, neg_snaps)
            picked_brick, picked_snap, polarity, direction = random.choice(
                visible_removable_snaps)
            
            if polarity == 0:
                picked_map = neg_snaps
            else:
                picked_map = pos_snaps
            
            snap_map = picked_map == [[[picked_brick, picked_snap]]]
            snap_map = snap_map[:,:,0] & snap_map[:,:,1]
            pick_ys, pick_xs = numpy.where(snap_map)
            
            pick_pixel = random.randint(0, len(pick_ys)-1)
            pick_y = pick_ys[pick_pixel]
            pick_x = pick_xs[pick_pixel]
            
            label = handspace_reassembly_template_action()
            label['disassembly']['activate'] = True
            label['disassembly']['polarity'] = polarity
            label['disassembly']['direction'] = direction
            label['disassembly']['pick'] = numpy.array(
                (pick_y, pick_x), dtype=numpy.long)
            labels.append(label)
        
        # if a brick was successfully disassembled, but incorrectly reassembled
        # then supervise something else in its place
        elif True:
            raise NotImplementedError # hold up cowboy
        
        # if a brick was successfully disassembled and reassembled,
        # then supervise the action taken
        elif True:
            raise NotImplementedError # no reassembly supervision yet
    
    # reassembly ===============================================================
    
    for i in range(first_reassembly_step, disassembling_steps.shape[0]):
        # at the moment, just end
        label = handspace_reassembly_template_action()
        #label['reassembly']['end'] = 1
        labels.append(label)
    
    if False:
        # figure out the order that the model was disassembled in
        disassembly_success_steps = numpy.where(
            disassembly_obs['success'] & disassembling_steps)
        disassembly_instance_ids = (
            disassembly_obs['instance_id'][disassembly_steps])
        
        # figure out which of the disassembled bricks were reassembled properly
        correctly_reassembled = set()
        # nothing was correctly reassembled if the reassembly phase hasn't begun
        if reassembling_steps[-1]:
            lkjlsdkjlksjf # TODO
        
        start_num_instances = workspace_configurations['num_instances'][0]
        incorrectly_reassembled = (
            set(range(1, start_num_instances+1)) - correctly_reassembled)
    
    return labels

def get_visible_removable_snaps(segmentation, pos_snaps, neg_snaps):
    visible_bricks = numpy.unique(segmentation)
    visible_bricks = visible_bricks[visible_bricks != 0]
    
    visible_pos_snaps = numpy.unique(pos_snaps.reshape(-1, 2), axis=0)
    visible_pos_snaps = visible_pos_snaps[visible_pos_snaps[:,0] != 0]
    visible_neg_snaps = numpy.unique(neg_snaps.reshape(-1, 2), axis=0)
    visible_neg_snaps = visible_neg_snaps[visible_neg_snaps[:,0] != 0]
    
    visible_removable_snaps = []
    for snaps, polarity in (visible_pos_snaps, 1), (visible_neg_snaps, 0):
        for brick_id, snap_id in snaps:
            visible_removable_snaps.append((brick_id, snap_id, polarity, 0))
            visible_removable_snaps.append((brick_id, snap_id, polarity, 1))
    
    return visible_removable_snaps
