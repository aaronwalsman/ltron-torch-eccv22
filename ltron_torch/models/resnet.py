import torch
import torchvision.models.resnet
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

class ResnetBackbone(torch.nn.Module):
    
    order = {
        'maxpool' : 0,
        'layer1' : 1,
        'layer2' : 2,
        'layer3' : 3,
        'layer4' : 4,
        'avgpool' : 5,
    }
    
    def __init__(self,
        resnet,
        *output_layers,
        frozen_weights=False,
        frozen_batchnorm=False,
    ):
        
        super(ResnetBackbone, self).__init__()
        self.resnet = resnet
        # remove the fc layer to free up memory
        del(self.resnet.fc)
        
        # TODO: Make output_layers a regular argument with a tuple default
        # instead of a *args
        if not len(output_layers):
            output_layers = list(self.order.keys())
        self.output_layers = output_layers
        # remove all unnecessary layers to free up memory
        for i, l in sorted([(v,k) for k,v in self.order.items()], reverse=True):
            self.last_layer = l
            if l in output_layers:
                break
            else:
                delattr(self.resnet, l)
        
        self.frozen_weights = frozen_weights
        if self.frozen_weights:
            for p in self.parameters():
                p.requires_grad = False
        
        self.frozen_batchnorm = frozen_batchnorm
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        f = {}
        f['maxpool'] = self.resnet.maxpool(x)
        
        if self.order[self.last_layer] >= self.order['layer1']:
            f['layer1'] = self.resnet.layer1(f['maxpool'])
        if self.order[self.last_layer] >= self.order['layer2']:
            f['layer2'] = self.resnet.layer2(f['layer1'])
        if self.order[self.last_layer] >= self.order['layer3']:
            f['layer3'] = self.resnet.layer3(f['layer2'])
        if self.order[self.last_layer] >= self.order['layer4']:
            f['layer4'] = self.resnet.layer4(f['layer3'])
        
        if self.order[self.last_layer] >= self.order['avgpool']:
            f['avgpool'] = self.resnet.avgpool(f['layer4'])
            f['avgpool'] = torch.flatten(f['avgpool'], 1)
        
        if not len(self.output_layers):
            return f
        else:
            return tuple(f[output_layer] for output_layer in self.output_layers)
    
    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)
        if self.frozen_batchnorm:
            for m in self.modules():
                if isinstance(m, (BatchNorm1d, BatchNorm2d, BatchNorm3d)):
                    m.eval()

def replace_fc(resnet, num_classes):
    fc = resnet.fc
    resnet.fc = torch.nn.Linear(
            fc.in_features, num_classes).to(fc.weight.device)

def replace_conv1(resnet, input_channels):
    conv1 = resnet.conv1
    resnet.conv1 = torch.nn.Conv2d(
            input_channels, conv1.out_channels,
            kernel_size=(7,7),
            stride=(2,2),
            padding=(3,3),
            bias=False).to(conv1.weight.device)

def named_backbone(name, *output_layers, pretrained=False, **kwargs):
    resnet = getattr(torchvision.models.resnet, name)(pretrained=pretrained)
    return ResnetBackbone(resnet, *output_layers, **kwargs)

def named_encoder_channels(name):
    if '18' in name or '34' in name:
        return (512, 256, 128, 64)
    elif '50' in name or '101' in name or '152' in name:
        return (2048, 1024, 512, 256)
    else:
        raise NotImplementedError
