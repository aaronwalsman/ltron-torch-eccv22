import torch
from torch.nn.functional import avg_pool2d

#from ltron_torch.models.simple_backbone import SimpleBackbone
import ltron_torch.models.resnet as resnet

class SimpleBlock(torch.nn.Module):
    def __init__(self, decoder_channels, skip_channels):
        super(SimpleBlock, self).__init__()
        self.skip_conv = torch.nn.Conv2d(
                skip_channels, decoder_channels, kernel_size=1)
    
    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            skip = self.skip_conv(skip)
            x = x + skip
        return x

class SimpleDecoder(torch.nn.Module):
    def __init__(self,
            encoder_channels,
            decoder_channels=256,
            decoder_depth=4):
        super(SimpleDecoder, self).__init__()
        
        layer = torch.nn.Conv2d(
                encoder_channels[0], decoder_channels, kernel_size=1)
        layers = [layer]
        for i in range(1, decoder_depth):
            layer = SimpleBlock(decoder_channels, encoder_channels[i])
            layers.append(layer)
        
        self.layers = torch.nn.ModuleList(layers)
    
    def forward(self, *features):
        x = self.layers[0](features[0])
        for layer, feature in zip(self.layers[1:], features[1:]):
            x = layer(x, feature)
        
        return x

class SimpleFCN(torch.nn.Module):
    def __init__(self,
        encoder,
        encoder_channels,
        decoder_channels,
        global_heads = None,
        dense_heads = None,
    ):
        super(SimpleFCN, self).__init__()
        self.encoder = encoder
        self.decoder = SimpleDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
        self.global_heads = global_heads
        self.dense_heads = dense_heads
    
    def forward(self, x):
        out = []
        xn = self.encoder(x)
        if self.global_heads is not None:
            xg = xn[0]
            b, c, h, w = xg.shape
            xg = avg_pool2d(xg, kernel_size=(h,w)).view(b,c)
            xg = self.global_heads(xg)
            out.append(xg)
        
        x = self.decoder(*xn)
        if self.dense_heads is not None:
            x = self.dense_heads(x)
        out.append(x)
        
        return tuple(out)

def named_resnet_fcn(
    name,
    decoder_channels,
    global_heads=None,
    dense_heads=None,
    pretrained=False,
    frozen_batchnorm=False,
):
    fcn_layers = ('layer4', 'layer3', 'layer2', 'layer1')
    encoder = resnet.named_backbone(
        name,
        *fcn_layers,
        pretrained=pretrained,
        frozen_batchnorm=frozen_batchnorm)
    encoder_channels = resnet.named_encoder_channels(name)
    return SimpleFCN(
        encoder,
        encoder_channels,
        decoder_channels,
        global_heads=global_heads,
        dense_heads=dense_heads,
    )

'''
class SimpleFCN(torch.nn.Module):
    def __init__(self,
        pretrained=True,
        decoder_channels=256,
        compute_single=True,
        backbone='resnet50',
    ):
        super(SimpleFCN, self).__init__()
        self.compute_single = compute_single
        if backbone == 'simple':
            backbone = SimpleBackbone()
            encoder_channels = (512, 256, 128, 64)
        elif backbone == 'resnet18':
            backbone = tv_models.resnet18(pretrained=pretrained)
            encoder_channels = (512, 256, 128, 64)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        elif backbone == 'resnet34':
            backbone = tv_models.resnet50(pretrained=pretrained)
            #encoder_channels = (512, 256, 128, 64)
            encoder_channels = (2048, 1024, 512, 256)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        elif backbone == 'resnet50':
            backbone = tv_models.resnet50(pretrained=pretrained)
            encoder_channels = (2048, 1024, 512, 256)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        self.encoder = backbone
        self.decoder = SimpleDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
    
    def forward(self, x):
        xn = self.encoder(x)
        x = self.decoder(*xn)
        if self.compute_single:
            x_single = torch.nn.functional.adaptive_avg_pool2d(xn[0], (1,1))
            x_single = torch.flatten(x_single, 1)
        else:
            x_single = None
        
        return x, x_single
'''
