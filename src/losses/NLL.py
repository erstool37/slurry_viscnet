import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.utils import loginterscaler, loginterdescaler, interscaler, interdescaler, zscaler, zdescaler, logzscaler, logzdescaler
import wandb

class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()

    def forward(self, mu, sigma, target):
        torch.clamp(mu, min=1e-9)
        nll = torch.log(sigma) + ((target[:,:3] - mu) ** 2) / (2 * sigma**2)

        loss_den = nll[:,0].mean()
        loss_visc = nll[:,1].mean()
        loss_surf = nll[:,2].mean()

        loss_total = loss_visc

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})  

        return loss_total
        