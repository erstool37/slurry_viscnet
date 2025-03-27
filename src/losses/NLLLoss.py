import torch
import torch.nn as nn
import torch.nn.functional as F

# this loss minimizes -log(p(viscosity|video)), asusming p must has normal distribution
class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        eps = 1e-8
        nll = torch.log(sigma) + ((target[:,:3] - mu) ** 2) / (2 * sigma**2)

        loss_den = nll[:,0].mean()
        loss_visc = nll[:,1].mean()
        loss_surf = nll[:,2].mean()

        print(nll[0])

        loss_total = loss_den + 3 * loss_visc + loss_surf

        return loss_total