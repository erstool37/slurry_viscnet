import torch
import torch.nn as nn
import wandb
import os.path as osp
import json
from src.utils.utils import loginterscaler, loginterdescaler, interscaler, interdescaler, zscaler, zdescaler, logzscaler, logzdescaler

class realMSLE(nn.Module):
    def __init__(self):
        super(realMSLE, self).__init__()

    def forward(self, pred, target):
        pred_den = loginterdescaler(pred[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynvisc = loginterdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        pred_surfT = loginterdescaler(pred[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        target_den = loginterdescaler(target[:,0], "density").unsqueeze(-1).to(pred.device)
        target_dynvisc = loginterdescaler(target[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        target_surfT = loginterdescaler(target[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        loss_den = torch.mean((torch.log1p(pred_den) - torch.log1p(target_den)) ** 2).unsqueeze(-1)
        loss_dynvisc = torch.mean((torch.log1p(pred_dynvisc) - torch.log1p(target_dynvisc)) ** 2).unsqueeze(-1)
        loss_surfT = torch.mean((torch.log1p(pred_surfT) - torch.log1p(target_surfT)) ** 2).unsqueeze(-1)

        total_loss = loss_den + 4000 * loss_dynvisc + 3 * loss_surfT

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})      

        return total_loss