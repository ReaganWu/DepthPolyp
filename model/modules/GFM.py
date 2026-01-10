import torch.nn as nn


class GFM_Module(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        init_channels = out_channels // ratio
        new_channels = out_channels - init_channels
        
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, 1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )
        
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # print("input:", x.shape)
        x1 = self.primary_conv(x)
        # print("primary conv output:", x1.shape)
        x2 = self.cheap_operation(x1)
        # print("cheap operation output:", x2.shape)
        return x1, x2