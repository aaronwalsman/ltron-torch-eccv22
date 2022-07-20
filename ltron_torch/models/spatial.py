import math

import torch

import ltron_torch.models.mlp as mlp

def nerf_position_encoding(x, dim=-1, L=10):
    encoding = []
    for l in range(L):
        xx = 2**l * math.pi * x
        encoding.append(torch.sin(xx))
        encoding.append(torch.cos(xx))
    
    encoding = torch.stack(encoding, dim)
    return encoding

class NerfSpatialEmbedding2D(torch.nn.Module):
    def __init__(self, num_layers, channels, L=10):
        super(NerfSpatialEmbedding2D, self).__init__()
        self.L = L
        self.mlp = mlp.Conv2dStack(num_layers, 4*self.L, channels, channels)
    
    def forward(self, x):
        height, width = x.shape[-2:]
        stack_shape = (1,2*self.L,height,width)
        
        yy = torch.linspace(-1, 1, height).to(x.device)
        yy = nerf_position_encoding(yy, dim=0, L=self.L)
        yy = yy.unsqueeze(0).unsqueeze(-1).expand(stack_shape)
        xx = torch.linspace(-1, 1, width).to(x.device)
        xx = nerf_position_encoding(xx, dim=0, L=self.L)
        xx = xx.unsqueeze(0).unsqueeze(-2).expand(stack_shape)
        
        yyxx = torch.cat((yy,xx), dim=1)
        
        x = x + self.mlp(yyxx)
        
        return x

class SpatialAttention2D(torch.nn.Module):
    def __init__(self, channels):
        super(SpatialAttention2D, self).__init__()
        self.attention_layer = torch.nn.Conv2d(channels, 1, kernel_size=1)
    
    def forward(self, x):
        attention = self.attention_layer(x)
        bs, _, h, w = attention.shape
        attention = torch.softmax(attention.view(bs, 1, -1), dim=-1)
        attention = attention.view(bs, 1, h, w)
        
        x = torch.sum(x * attention, dim=(2,3))
        
        return x

class SE3Layer(torch.nn.Module):
    def __init__(self, dim=-1, orientation_axes='YZX', translation_scale=1.):
        super(SE3Layer, self).__init__()
        self.dim = dim
        self.orientation_axes = orientation_axes
        self.translation_scale = translation_scale
    
    def forward(self, x):
        primary_axis_indices = [slice(None) for _ in x.shape]
        primary_axis_indices[self.dim] = slice(0,3)
        
        secondary_axis_indices = [slice(None) for _ in x.shape]
        secondary_axis_indices[self.dim] = slice(3,6)
        
        translation_indices = [slice(None) for _ in x.shape]
        translation_indices[self.dim] = slice(6,9)
        
        primary_axis = x[tuple(primary_axis_indices)]
        primary_axis = (
            primary_axis / torch.norm(primary_axis, dim=self.dim, keepdim=True))
        secondary_axis = x[tuple(secondary_axis_indices)]
        third_axis = torch.cross(primary_axis, secondary_axis, dim=self.dim)
        third_axis = (
            third_axis / torch.norm(third_axis, dim=self.dim, keepdim=True))
        if self.orientation_axes in ('XZY', 'YXZ', 'ZYX'):
            flip = -1.
        else:
            flip = 1.
        third_axis = third_axis * flip
        secondary_axis = torch.cross(third_axis, primary_axis)
        secondary_axis = secondary_axis * flip
        
        translation = x[tuple(translation_indices)] / self.translation_scale
        
        indices = {'X':0, 'Y':1, 'Z':2}
        axes = [None, None, None]
        
        if self.dim >= 0:
            i = self.dim+1
        else:
            i = self.dim
        axes[indices[self.orientation_axes[0]]] = primary_axis.unsqueeze(i)
        axes[indices[self.orientation_axes[1]]] = secondary_axis.unsqueeze(i)
        axes[indices[self.orientation_axes[2]]] = third_axis.unsqueeze(i)
        #axes.append(translation.unsqueeze(i))
        
        rotation = torch.cat(axes, dim=i)
        
        return {'rotation':rotation, 'translation':translation}

class SelfAttentionPose(torch.nn.Module):
    def __init__(self, in_features, stack_layers=3, translation_scale=1.):
        super(SelfAttentionPose, self).__init__()
        self.pose_stack = mlp.LinearStack(
            stack_layers,
            in_features = in_features*2,
            hidden_features = in_features,
            out_features = 9,
        )
        self.pose_layer = SE3Layer(translation_scale=translation_scale)
    
    def forward(self, x):
        s, b, c = x.shape
        x = mlp.cross_product_concat(x, x)
        x = self.pose_stack(x)
        x = self.pose_layer(x)
        return x

