import torch
import numpy as np


def Dice_Metric(predict, target):
    # print(f"predict: {predict.shape}, target: {target.shape}")
    intersection = 2.0 * (target*predict).sum()
    union = target.sum() + predict.sum()
    if target.sum() == 0 and predict.sum() == 0:
        return 1.0 
    return intersection/union

def Jaccard_Index(predict, target):
    intersection = (predict * target).sum()
    union = predict.sum() + target.sum() - intersection
    if union == 0:
        return 1.0  # 当分子和分母都为0时，Jaccard指标为1
    else:
        return intersection / union

def Recall(predict, target):
    true_positives = (predict * target).sum()
    false_negatives = target.sum() - true_positives
    if true_positives + false_negatives == 0:
        return 1.0  # 当真正例和假反例都为0时，召回率为1
    else:
        return true_positives / (true_positives + false_negatives)
