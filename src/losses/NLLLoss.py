import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.PreprocessorPara import logdescaler, zdescaler

# this loss minimizes -log(p(viscosity|video)), asusming p must has normal distribution
class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        nll = torch.log(sigma) + ((target[:,:3] - mu) ** 2) / (2 * sigma**2)

        loss_den = nll[:,0].mean()
        loss_visc = nll[:,1].mean()
        loss_surf = nll[:,2].mean() 

        loss_total = loss_den + 10**3 * loss_visc + loss_surf

        # MAPE calculation
        pred_den = logdescaler(mu[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynvisc = logdescaler(mu[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        pred_surfT = logdescaler(mu[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        target_den = logdescaler(target[:,0], "density").unsqueeze(-1).to(pred.device)
        target_dynvisc = logdescaler(target[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        target_surfT = logdescaler(target[:,2], "surface_tension").unsqueeze(-1).to(pred.device)

        loss_mape_den = torch.mean((torch.abs(pred_den - target_den) / target_den)).unsqueeze(-1)
        loss_mape_dynvisc = torch.mean((torch.abs(pred_dynvisc - target_dynvisc) / target_dynvisc)).unsqueeze(-1)
        loss_mape_surfT = torch.mean((torch.abs(pred_surfT - target_surfT) / target_surfT)).unsqueeze(-1)

        wandb.log({"MAPE den %" : loss_mape_den * 100})
        wandb.log({"MAPE dynvisc %" : loss_mape_dynvisc * 100})
        wandb.log({"MAPE surfT %" : loss_mape_surfT * 100})

        return loss_total