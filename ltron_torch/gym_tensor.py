import torch
import torchvision.transforms as transforms

import numpy

import gym.spaces as spaces

from ltron.gym.spaces import (
        ImageSpace, SegmentationSpace, StepSpace, ClassLabelSpace,
        EdgeSpace, InstanceGraphSpace)
#from ltron.torch.brick_geometric import (
#        BrickList, BrickGraph, BrickListBatch, BrickGraphBatch)

default_mean = [0.485, 0.456, 0.406]
default_std = [0.229, 0.224, 0.225]

default_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=default_mean, std=default_std),
])

tensor_mean = torch.FloatTensor(default_mean)
tensor_std = torch.FloatTensor(default_std)

default_tile_transform = transforms.Compose([
    lambda x : torch.FloatTensor(x)/255.,
    lambda x : (x - tensor_mean.view(1,1,1,3)) / tensor_std.view(1,1,1,3)
])

def default_image_untransform(x):
    device = x.device
    mean = torch.FloatTensor(default_mean).unsqueeze(-1).unsqueeze(-1)
    mean = mean.to(device)
    std = torch.FloatTensor(default_std).unsqueeze(-1).unsqueeze(-1)
    std = std.to(device)
    x = x * std + mean
    x = (x * 255).byte().cpu().numpy()
    x = numpy.moveaxis(x,-3,-1)
    return x

def gym_space_to_tensors(
        data,
        space,
        device=torch.device('cpu'),
        image_transform=default_image_transform):
    def recurse(data, space):
        if isinstance(space, ImageSpace):
            if len(data.shape) == 3:
                tensor = image_transform(data)
            elif len(data.shape) == 4:
                tensor = torch.stack(
                        tuple(image_transform(image) for image in data))
            return tensor.to(device)
        elif isinstance(space, SegmentationSpace):
            return torch.LongTensor(data).to(device)
        
        elif isinstance(space, StepSpace):
            return torch.LongTensor(data).to(device)
        
        #elif isinstance(space, ClassLabelSpace):
        #    return {
        #        'label':torch.LongTensor(data['label']).to(device),
        #        'num_instances':torch.LongTensor(
        #            data['num_instances']).to(device)
        #    }
        
        elif isinstance(space, EdgeSpace):
            # TODO: SCORE
            tensor_dict = {
                    'edge_index' :
                        torch.LongTensor(data['edge_index']).to(device)
            }
            return tensor_dict
        
        elif isinstance(space, InstanceGraphSpace):
            brick_list = recurse(data['instances'], space['instances'])
            edge_tensor_dict = recurse(data['edges'], space['edges'])
            edge_index = edge_tensor_dict['edge_index']
            # TODO: SCORE
            if isinstance(brick_list, BrickList):
                return BrickGraph(brick_list, edge_index=edge_index)
            elif isinstance(brick_list, BrickListBatch):
                return BrickGraphBatch.from_brick_list_batch(
                        brick_list, edge_index)
        
        # keep the default spaces last because ltron's custom spaces
        # inherit from them so those cases should be caught first
        elif isinstance(space, spaces.Discrete):
            return torch.LongTensor(data).to(device)
        
        elif isinstance(space, spaces.Box):
            #return torch.FloatTensor(data).to(device)
            return torch.as_tensor(data, device=device)
        
        elif isinstance(space, spaces.Dict):
            return {key : recurse(data[key], space[key]) for key in data}
        
        elif isinstance(space, spaces.Tuple):
            return tuple(recurse(d, s) for d,s in zip(data, space))
    
    return recurse(data, space)

def gym_space_list_to_tensors(
        data,
        space,
        device=torch.device('cpu'),
        image_transform=default_image_transform):
    '''
    Everything added here should be set up so that if it already has a batch
    dimension, that should become the PRIMARY dimension (dimension 0) so that
    when everything is viewed as one long list, neighboring entries in the
    batch dimension come from the same sequence.
    '''
    tensors = [gym_space_to_tensors(d, space, device, image_transform)
            for d in data]
    
    def recurse(data, space):
        if isinstance(space, ImageSpace):
            c, h, w = data[0].shape[-3:]
            tensor = torch.stack(data, dim=-4)
            return tensor.view(-1, c, h, w)
        
        elif isinstance(space, SegmentationSpace):
            h, w = data[0].shape[-2:]
            tensor = torch.stack(data, dim=-3)
            return tensor.view(-1, h, w)
        
        elif isinstance(space, StepSpace):
            c = data[0].shape[-1]
            tensor = torch.stack(data, dim=-2)
            return tensor.view(-1, c)
        
        #elif isinstance(space, InstanceGraphSpace):
        #    return BrickGraphBatch.join(data, transpose=True)
        
        elif isinstance(space, spaces.Dict):
            return {key : recurse([d[key] for d in data], space[key])
                for key in data[0]}
        
        elif isinstance(space, spaces.Tuple):
            return tuple(recurse(data[i], space[i]) for i in len(data[0]))
        
        elif isinstance(space, spaces.Box):
            #c = data[0].shape[-1:]
            c = data[0].shape[1:]
            tensor = torch.stack(data, dim=1)
            return tensor.view(-1, *c)
        
        elif isinstance(space, spaces.Discrete):
            c = data[0].shape[1:]
            tensor = torch.stack(data, dim=1)
            return tensor.view(-1, *c)
    
    return recurse(tensors, space)

def graph_to_gym_space(
        data,
        space,
        process_instance_logits=False,
        segment_id_remap=False):
    
    # build the instance labels
    instance_labels = numpy.zeros(
            (space['instances']['label'].shape[0]), dtype=numpy.long)
    instance_scores = numpy.zeros(
            (space['instances']['score'].shape[0]))
    
    edge_index = data['edge_index'].detach().cpu().numpy()
    
    num_instances = data.num_nodes
    if num_instances:
        
        # process logits
        if process_instance_logits:
            # discretize the labels
            discrete_labels = torch.argmax(data['instance_label'], dim=-1)
            discrete_labels = discrete_labels.detach().cpu().numpy()
        else:
            discrete_labels = data['instance_label'].detach().cpu().numpy()
        
        # remap
        if segment_id_remap:
            # labels
            segment_id = data['segment_id'].view(-1).detach().cpu().numpy()
            num_instances = int(numpy.max(segment_id))
            instance_indices = segment_id
            
            # In the case where multiple nodes have the same segment_id, it
            # may be better to do a scatter-max based on score than what we're
            # doing here.  As it is, this is defaulting to numpy behavior when
            # the same index is specified twice during __setitem__ which I
            # think is to take the last element specified.
            
            # edges
            edge_index = segment_id[edge_index]
        
        else:
            instance_indices = list(range(num_instances))
        
        # make sure there aren't too many labels
        assert num_instances <= space['instances'].max_instances
        
        # plug the data into the numpy tensors
        instance_labels[instance_indices] = discrete_labels
        instance_labels = instance_labels.reshape(-1, 1)
        if space['instances'].include_score:
            instance_scores[instance_indices] = (
                    data['score'].detach().cpu().view(-1).numpy())
            instance_scores = instance_scores.reshape(-1, 1)
    
    # compile the instance data
    instance_data = {
            'num_instances' : num_instances,
            'label' : instance_labels,
    }
    if space['instances'].include_score:
        instance_data['score'] = instance_scores
    
    edge_data = {
            'num_edges' : data.num_edges(),
            'edge_index' : edge_index,
    }
    if space['edges'].include_score:
        edge_data['score'] = data['edge_attr'].cpu().numpy()
    
    # compile result
    result = {
            'instances' : instance_data,
            'edges' : edge_data,
    }
    #if 'edge_scores' in space.spaces:
    #    edge_score = data['edge_attr'][:,0].detach().cpu().numpy()
    #    result['edge_scores'] = edge_score
    
    return result
