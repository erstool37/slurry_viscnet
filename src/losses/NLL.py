import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import importlib

class NLL(nn.Module):
    """ 
    Negative Log Likelihood Loss, assuming Gaussian distribution output. plz use BayesianViscosityEstimator for this loss
    """
    def __init__(self, unnormalizer, path):
        super(NLL, self).__init__()

    def forward(self, mu, sigma, target):
        torch.clamp(mu, min=1e-6, max=10)
        torch.clamp(sigma, min=1e-6, max=10)
        nll = torch.log(sigma) + ((target[:,:3] - mu) ** 2) / (2 * sigma**2)
 
        loss_den = nll[:,0].mean()
        loss_visc = nll[:,1].mean()
        loss_surfT = nll[:,2].mean()

        loss_total = loss_visc

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_visc})
        wandb.log({"loss_surf": loss_surfT})  

        return loss_total
        