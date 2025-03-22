import torch
import torch.nn as nn
import torch.nn.functional as F

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)  # Ensure no negative predictions
        target = torch.clamp(target, min=0)

        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2
        return loss.mean()
