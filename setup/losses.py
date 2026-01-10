import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten predictions and ground truth for Dice computation
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute intersection between prediction and target
        intersection = (inputs * targets).sum()

        # Dice coefficient
        dice = (2. * intersection + smooth) / \
               (inputs.sum() + targets.sum() + smooth)

        # Dice loss
        return 1 - dice


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()

        # Learnable uncertainty parameters for each task
        self.sigma1 = nn.Parameter(torch.ones(1))  # uncertainty for Dice loss (segmentation)
        self.sigma2 = nn.Parameter(torch.ones(1))  # uncertainty for SmoothL1 loss (depth)

        # Loss functions for individual tasks
        self.dice_loss = DiceLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred_seg, target_seg, pred_depth, target_depth):
        # Compute task-specific losses
        dice_loss = self.dice_loss(pred_seg, target_seg)
        depth_loss = self.smooth_l1_loss(pred_depth, target_depth)

        # Uncertainty-weighted multi-task loss with regularization term
        total_loss = (
            (1 / (self.sigma1 ** 2)) * dice_loss +
            (1 / (self.sigma2 ** 2)) * depth_loss +
            torch.log(self.sigma1 * self.sigma2)
        )

        return total_loss
