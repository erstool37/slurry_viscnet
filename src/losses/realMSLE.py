import torch
import torch.nn as nn
import wandb
import os.path as osp
import json
import importlib

class realMSLE(nn.Module):
    """
    unnormalized into real scale and MSLE calculated
    """
    def __init__(self, unnormalizer, path):
        super(realMSLE, self).__init__()
        self.unnormalizer = unnormalizer
        self.path = path

    def forward(self, pred, target):
        utils = importlib.import_module("utils")
        descaler = getattr(utils, self.unnormalizer)

        pred_den = descaler(pred[:,0], "density", self.path).unsqueeze(-1).to(pred.device)
        pred_dynvisc = descaler(pred[:,1], "dynamic_viscosity", self.path).unsqueeze(-1).to(pred.device)
        pred_surfT = descaler(pred[:,2], "surface_tension", self.path).unsqueeze(-1).to(pred.device)

        target_den = descaler(target[:,0], "density", self.path).unsqueeze(-1).to(pred.device)
        target_dynvisc = descaler(target[:,1], "dynamic_viscosity", self.path).unsqueeze(-1).to(pred.device)
        target_surfT = descaler(target[:,2], "surface_tension", self.path).unsqueeze(-1).to(pred.device)

        loss_den = torch.mean((torch.log1p(pred_den) - torch.log1p(target_den)) ** 2).unsqueeze(-1)
        loss_dynvisc = torch.mean((torch.log1p(pred_dynvisc) - torch.log1p(target_dynvisc)) ** 2).unsqueeze(-1)
        loss_surfT = torch.mean((torch.log1p(pred_surfT) - torch.log1p(target_surfT)) ** 2).unsqueeze(-1)

        total_loss = loss_dynvisc
        print("pred_den", pred_den)
        print("target_den", target_den)
        print("loss_den", loss_den)
        print("pred_dynvisc", pred_dynvisc)
        print("target_dynvisc", target_dynvisc)
        print("loss_dynvisc", loss_dynvisc)
        print("pred_surfT", pred_surfT)
        print("target_surfT", target_surfT)
        print("loss_surfT", loss_surfT)
        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})      

        return total_loss