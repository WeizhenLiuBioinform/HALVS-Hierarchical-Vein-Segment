import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F


def exclusive_loss(pred, exclude_label=None, mode='entropy'):
    if exclude_label is None:
        exclude_label = [1, 2]
    if mode == 'entropy':
        label = torch.zeros_like(pred)
        for i in exclude_label:
            label[:, i, :, :] = 1      # [0,1,1,0]
        pred_softmax = pred.softmax(dim=1)
        pred_exclusive_entropy = torch.log(1 + pred_softmax)
        loss = label * pred_exclusive_entropy
        return loss