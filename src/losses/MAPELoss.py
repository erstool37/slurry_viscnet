import torch
import torch.nn as nn
from src.utils.PreprocessorPara import logdescaler, zdescaler
import wandb
import os.path as osp
import json

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, pred, target):
        # MAPE module
        pred_den = logdescaler(pred[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynvisc = logdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        pred_surfT = logdescaler(pred[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        target_den = logdescaler(target[:,0], "density").unsqueeze(-1).to(pred.device)
        target_dynvisc = logdescaler(target[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        target_surfT = logdescaler(target[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
        loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
        loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

        # wandb.log({"MAPE den %" : loss_mape_den * 100})
        wandb.log({"MAPE dynvisc %" : loss_mape_dynvisc * 100})
        # wandb.log({"MAPE surfT %" : loss_mape_surfT * 100})

        # MSLE calculation
        # loss_den = torch.mean((torch.log1p(pred_den) - torch.log1p(target_den)) ** 2).unsqueeze(-1)
        # loss_dynvisc = torch.mean((torch.log1p(pred_dynvisc) - torch.log1p(target_dynvisc)) ** 2).unsqueeze(-1)
        # loss_surfT = torch.mean((torch.log1p(pred_surfT) - torch.log1p(target_surfT)) ** 2).unsqueeze(-1)

        # wandb.log({"loss chunk den %" : loss_den})
        # wandb.log({"loss chunk dynvisc %" : 4000 * loss_dynvisc})
        # wandb.log({"loss chunk surfT %" : 3 * loss_surfT})

        # total_loss = loss_den + 4000 * loss_dynvisc + 3 * loss_surfT
        total_loss = loss_mape_dynvisc
        
        """
        path = osp.dirname(osp.abspath(__file__))
        stat_path = osp.join(path, "../../dataset/CFDfluid/statistics.json")
        with open(stat_path, 'r') as file:
            data = json.load(file)
            mean_den = torch.tensor(data["density"]["mean"], dtype=pred.dtype, device=pred.device)
            mean_visc = torch.tensor(data["dynamic_viscosity"]["mean"], dtype=pred.dtype, device=pred.device)
            mean_surfT = torch.tensor(data["surface_tension"]["mean"], dtype=pred.dtype, device=pred.device)

        target_den = logdescaler(target[:,0], "density").unsqueeze(-1).to(pred.device)
        target_dynvisc = logdescaler(target[:,1], "density").unsqueeze(-1).to(pred.device)
        target_surfT = logdescaler(target[:,2], "density").unsqueeze(-1).to(pred.device)

        loss = (pred - target[:,:3] / target[:,:3])**2
        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()
        total_loss = loss_den + loss_dynvisc + loss_surfT

        loss_real_den = torch.mean(loss[:,0] ** 0.5 / target_den * (target_den - mean_den))
        loss_real_dynvisc = torch.mean(loss[:,1] ** 0.5 / target_dynvisc * (target_dynvisc - mean_visc))
        loss_real_surfT = torch.mean(loss[:,2] ** 0.5 / target_surfT * (target_surfT- mean_surfT))

        wandb.log({"MAPE den %" : loss_real_den * 100})
        wandb.log({"MAPE dynvisc %" : loss_real_dynvisc * 100})
        wandb.log({"MAPE surfT %" : loss_real_surfT * 100})
        """

        return total_loss