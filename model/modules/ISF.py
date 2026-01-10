import torch
import torch.nn as nn

class GroupChannelShuffle(nn.Module):
    """
    group-based channel shuffle / interleave.
    groups: number of source groups you want to interleave (e.g. 4 for c1..c4)
    optional cyclic shift (percent) to add deterministic rotation after shuffle.
    """
    def __init__(self, groups: int = 4, cyclic_percent: float = 0.0):
        super().__init__()
        assert groups >= 1
        self.groups = groups
        self.cyclic_percent = cyclic_percent

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        g = self.groups
        assert C % g == 0, f"channels {C} not divisible by groups {g}"
        gc = C // g
        # reshape to (B, groups, group_channels, H, W)
        x = x.view(B, g, gc, H, W)
        # transpose to interleave: (B, group_channels, groups, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)
        # optional cyclic rotate by percent of channels (deterministic)
        if self.cyclic_percent and 0 < self.cyclic_percent < 1.0:
            shift = int(C * self.cyclic_percent)
            x = torch.roll(x, shifts=shift, dims=1)
        return x

class ISF_Module(nn.Module):
    """
    A lightweight module that wraps shuffle + depthwise conv + group-wise scaling + residual.
    - channels: total channels of x
    - groups: number of logical groups (must divide channels)
    """
    def __init__(self, channels: int, groups: int = 4, kernel_size: int = 3, cyclic_percent: float = 0.0):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.channels = channels
        self.shuffle = GroupChannelShuffle(groups=groups, cyclic_percent=cyclic_percent)

        # depthwise conv (per-channel local spatial enhancement)
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        # group-wise scaling: one scalar per group to reweight groups after fusion
        self.group_scale = nn.Parameter(torch.ones(groups), requires_grad=True)  # tiny param overhead

        # optional small pointwise to re-calibrate channels (commented out to keep ultra-light)
        # self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 1) deterministic interleave
        y = self.shuffle(x)               # (B, C, H, W)

        # 2) per-channel spatial refine
        y = self.dw(y)
        y = self.bn(y)
        y = self.act(y)

        # 3) group-wise scaling
        gc = C // self.groups
        # scale = self.group_scale.repeat_interleave(gc).view(1, C, 1, 1)  # (1, C, 1, 1)
        scale = self.group_scale.to(x.device)  
        scale = scale.repeat_interleave(gc).view(1, C, 1, 1)
        y = y * scale

        # 4) residual add to preserve original information
        out = x + y
        return out

