import torch
import torch.nn as nn
import torch.nn.functional as F

class RealNVP(nn.Module):
    """
    Conditional RealNVP model for normalizing flows(non volume preserving-affine fcn, different from NICE which is volume preserving-addition).
    takes conditions(video) and the predicted viscosity, and maps its PDF to a standard normal distribution using a series of affine transformations.
    """
    def __init__(self, dim, cond_dim, hidden_dim, num_layers):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.num_layers = num_layers

        self.masks = [self._create_mask(i) for i in range(num_layers)]
        self.STnet = nn.ModuleList([self._build_STnet(dim, cond_dim, hidden_dim) for _ in range(num_layers)])

    def _create_mask(self, layer_idx): # masking
        mask = torch.zeros(self.dim)
        mask[layer_idx % 2::2] = 1 # odd/even layer masked every layer
        return mask

    def _build_STnet(self, dim, cond_dim, hidden_dim):
        nlp = nn.Sequential(
                nn.Linear(dim // 2 + cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
                )
        return nlp

    def forward(self, y, c):
        log_det = 0
        x = y
        for mask, stnet in zip(self.masks, self.STnet):
            mask = mask.to(x.device)
            # mask some part of x
            x_masked = x * mask
            input_st = torch.cat([x_masked * (1 - mask), c], dim=1)

            # pass the unmasked part into STnet to get s and t
            st = stnet(input_st)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s) # for stability

            # apply the affine transformation
            x = x_masked + (1 - mask) * (x * torch.exp(s) + t) # makes the Jacobian triangular, making det calculation easier
            log_det += ((1 - mask) * s).sum(dim=1)
        return x, log_det

    def inverse(self, z, c):
        x = z
        for mask, stnet in reversed(list(zip(self.masks, self.st_nets))):
            mask = mask.to(x.device)
            x_masked = x * mask
            input_st = torch.cat([x_masked * (1 - mask), c], dim=1)
            st = stnet(input_st)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            x = x_masked + (1 - mask) * (x - t) * torch.exp(-s)
        return x