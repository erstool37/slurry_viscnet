import torch
import torch.nn as nn
import wandb

class MSLELoss(nn.Module):
    def __init__(self, model):
        super(MSLELoss, self).__init__()
        self.model = model

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=0)
        target = torch.clamp(target, min=0)
        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2 

        loss_den = loss[:, 0].mean()
        loss_dynvisc = loss[:, 1].mean()
        loss_surfT = loss[:, 2].mean()

        loss_total = loss_den + loss_dynvisc + loss_surfT
        # loss_total = loss_den * 869.831735393482 +  * 10**(loss_dynvisc * 3.657577315785776-0.029653116674823305) + loss_surfT * 0.033626156085620175

        wandb.log({"loss_den": loss_den})
        wandb.log({"loss_visc": loss_dynvisc})
        wandb.log({"loss_surf": loss_surfT})
        # wandb.log({"loss_den_chunk": loss_den * 869.831735393482})
        # wandb.log({"loss_visc_chunk": loss_dynvisc * 10**(3.657577315785776-0.029653116674823305)})
        # wandb.log({"loss_surf_chunk": loss_surfT * 0.033626156085620175})
        
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
        pred = torch.clamp(pred, min=0)  # Ensure no negative predictions
        target = torch.clamp(target, min=0)

        loss = (torch.log1p(pred) - torch.log1p(target)) ** 2
        return loss.mean()
    """