import torch
import torch.nn as nn
import wandb
import os.path as osp
import json
from src.utils.utils import loginterscaler, loginterdescaler, interscaler, interdescaler, zscaler, zdescaler, logzscaler, logzdescaler

class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()

    def forward(self, pred, target):
        pred_den = loginterdescaler(pred[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynvisc = loginterdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        pred_surfT = loginterdescaler(pred[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        target_den = loginterdescaler(target[:,0], "density").unsqueeze(-1).to(pred.device)
        target_dynvisc = loginterdescaler(target[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        target_surfT = loginterdescaler(target[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
        loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
        loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

        total_loss = loss_mape_dynvisc
        
        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})      

        return total_loss