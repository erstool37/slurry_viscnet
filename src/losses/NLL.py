import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import importlib

class NLL(nn.Module):
    """ 
    Negative Log Likelihood Loss, Maximizing posterior assuming Gaussian distribution output, and without prior
    use flow model for this loss
    """
    def __init__(self, unnormalizer, path):
        super(NLL, self).__init__()

    def forward(self, z, log_det_jacobian):
        log_prob = -0.5 * (z ** 2 + torch.log(torch.tensor(2 * torch.pi, device=z.device)))
        nll = -log_prob - log_det_jacobian.unsqueeze(1)  # shape: [batch, dim]

        loss_den = nll[:,0].mean()
        loss_visc = nll[:,1].mean()
        loss_surfT = nll[:,2].mean()

        loss_total = loss_den + loss_visc + loss_surfT

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_visc})
        wandb.log({"loss_surf": loss_surfT})
        
        return loss_total