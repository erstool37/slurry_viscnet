import torch
import torch.nn as nn
import wandb
from src.utils.PreprocessorPara import logdescaler, zdescaler
import torch.nn.functional as F

class MSLELoss(nn.Module):
    def __init__(self, model):
        super(MSLELoss, self).__init__()
        self.model = model

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)

        # predicted kinVisc calculation in unnormalized version
        pred_den = logdescaler(pred[:,0], "density").unsqueeze(-1).to(pred.device)
        pred_dynVisc = logdescaler(pred[:,1], "dynamic_viscosity").unsqueeze(-1).to(pred.device)
        
        pred_kinVisc = (pred_dynVisc / pred_den)

        pred = torch.cat((pred, pred_kinVisc), dim=-1)

        # real kinVisc calculation
        ans_kinVisc = logdescaler(pred[:,3], "kinematic_viscosity")
        target[:,3] = ans_kinVisc

        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2 
        # loss = (pred-target) ** 2

        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()
        loss_kinvisc = loss[:, 3].mean()
        # loss_kinvisc = F.mse_loss(pred[:,3], target[:,3]) # already mean

        loss_total = 5 * loss_den +  1 * loss_surfT + 5 * loss_dynvisc
        # loss_total = 1 * loss_den +  1 * loss_surfT + 10**4 * loss_kinvisc

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})
        # wandb.log({"loss_kinvisc": loss_kinvisc})

        wandb.log({"loss_den_chunk": 5 * loss_dynvisc})
        wandb.log({"loss_visc_chunk": 5 * loss_dynvisc})
        wandb.log({"loss_surf_chunk": 1 * loss_surfT})
        # wandb.log({"loss_kinvisc_chunk": 10**3 * loss_kinvisc})
        
        return loss_total

    # this is adaptive constant method code
    """
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)

        # Compute MSLE loss
        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2 

        # Separate loss terms for each variable
        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()

        # Compute gradients for weighting
        # gradden = torch.autograd.grad(loss_den, self.model.parameters(), retain_graph=True, create_graph=True)
        gradvisc = torch.autograd.grad(loss_dynvisc, self.model.parameters(), retain_graph=True, create_graph=True)
        gradsurf = torch.autograd.grad(loss_surfT, self.model.parameters(), retain_graph=True, create_graph=True)

        # Compute gradient-based weights
        weight_den = 1.0
        weight_visc = torch.norm(torch.cat([g.flatten() for g in grad1])) / (torch.norm(torch.cat([g.flatten() for g in gradvisc])) + 1e-8)
        weight_surf = torch.norm(torch.cat([g.flatten() for g in grad1])) / (torch.norm(torch.cat([g.flatten() for g in gradsurf])) + 1e-8)

        # Total loss (weighted sum)
        loss_total = weight_den * loss_den + weight_visc * loss_dynvisc + weight_surf * loss_surfT

        # Log losses in wandb
        wandb.log({"loss_den": loss_den.item()})
        wandb.log({"loss_visc": loss_dynvisc.item()})
        wandb.log({"loss_surf": loss_surfT.item()})
        wandb.log({"loss_den_chunk": (weight_den * loss_den).item()})
        wandb.log({"loss_visc_chunk": (weight_visc * loss_dynvisc).item()})
        wandb.log({"loss_surf_chunk": (weight_surf * loss_surfT).item()})
        
        # Return total loss for backpropagation
        return loss_total
    """

    # simple msle loss method
    """
    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2 

        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()

        loss_total = loss_den + loss_dynvisc + loss_surfT

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})
        return loss_total
    """