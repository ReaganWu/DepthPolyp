import torch.nn as nn


# ============================================================================
# Activation Module
# ============================================================================

class Activation(nn.Module):
    """Activation wrapper that supports various activation functions"""
    def __init__(self, activation=None):
        super().__init__()
        
        if activation is None or activation == 'identity':
            self.activation = nn.Identity()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'softmax2d':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'logsoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif callable(activation):
            self.activation = activation
        else:
            raise ValueError(
                f'Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {activation}'
            )
    
    def forward(self, x):
        return self.activation(x)
    
# ============================================================================
# Segmentation Head (nn.Sequential style)
# ============================================================================

class SegmentationHead(nn.Sequential):
    """Segmentation head using nn.Sequential style"""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=None,
        upsampling=1
    ):
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        upsampling_layer = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling) 
            if upsampling > 1 
            else nn.Identity()
        )
        activation_layer = Activation(activation)
        super().__init__(conv2d, upsampling_layer, activation_layer)
