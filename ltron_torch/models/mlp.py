import torch

def stack(
    Layer,
    num_layers,
    in_channels,
    hidden_channels,
    out_channels,
    norm=None,
    nonlinearity=torch.nn.ReLU,
    **kwargs,
):
    try:
        hidden_channels = [int(hidden_channels)] * (num_layers-1)
    except TypeError:
        assert len(hidden_channels) == num_layers-1
    
    features = [in_channels] + hidden_channels + [out_channels]
    
    layers = []
    for i, (in_f, out_f) in enumerate(zip(features[:-1], features[1:])):
        layers.append(Layer(in_f, out_f, **kwargs))
        
        if i != num_layers-1:
            if norm is not None:
                layers.append(norm())
            if nonlinearity is not None:
                layers.append(nonlinearity())
    
    return torch.nn.Sequential(*layers)

def linear_stack(*args, **kwargs):
    return stack(torch.nn.Linear, *args, **kwargs)

def conv2d_stack(*args, kernel_size=1, **kwargs):
    return stack(torch.nn.Conv2d, *args, kernel_size=kernel_size, **kwargs)

def cross_product_concat(xa, xb):
    sa, ba, ca = xa.shape
    sb, bb, cb = xb.shape
    assert ba == bb
    
    xa = xa.view(sa, 1, ba, ca).expand(sa, sb, ba, ca)
    xb = xb.view(1, sb, bb, cb).expand(sa, sb, bb, cb)
    x = torch.cat((xa, xb), dim=-1)
    
    return x

class Conv2dStack(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_features,
        hidden_features,
        out_features,
    ):
        super(Conv2dStack, self).__init__()
        layer_features = (
            [in_features] + [hidden_features] * (num_layers-1) + [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Conv2d(in_f, out_f, 1))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
    
    def forward(self, x):
        return self.layers(x)

class LinearStack(torch.nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features):
        super(LinearStack, self).__init__()
        layer_features = (
                [in_features] +
                [hidden_features] * (num_layers-1) +
                [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Linear(in_f, out_f))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.layers(x)

