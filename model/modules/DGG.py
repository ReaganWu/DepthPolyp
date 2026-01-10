import torch
import torch.nn as nn


class DGG_Module(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.groups = groups
        self.fc = nn.Linear(groups, groups)

    def forward(self, x):
        B, C, H, W = x.shape
        gc = C // self.groups

        xg = x.view(B, self.groups, gc, H, W).mean(dim=(2,3,4))  # (B, groups)
        gates = torch.sigmoid(self.fc(xg))[:, :, None, None, None]  # (B, groups, 1, 1, 1)

        xg = x.view(B, self.groups, gc, H, W)
        out = (xg * gates).reshape(B, C, H, W)
        return out
