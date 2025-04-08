import torch
import torch.nn as nn
import torch.nn.functional as F

class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim, hidden_layers):
        super().__init__()
        self.dim = dim
        self.masks = [self._get_mask(i) for i in range(hidden_layers)]
        self.s_t_layers = nn.ModuleList([self._build_st_network(dim, hidden_dim) for _ in range(hidden_layers)])

    def _get_mask(self, layer_idx):
        mask = torch.zeros(self.dim)
        mask[layer_idx % 2::2] = 1
        return mask

    def _build_st_network(self, dim, hidden_dim):
        mlp =  nn.Sequential(
                nn.Linear(dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
                )
        return mlp

    def forward(self, x):
        log_det_jacobian = 0
        for mask, st_net in zip(self.masks, self.s_t_layers):
            mask = mask.to(x.device)
            x_masked = x * mask
            st = st_net(x_masked * (1 - mask))
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            x = x_masked + (1 - mask) * (x * torch.exp(s) + t)
            log_det_jacobian += ((1 - mask) * s).sum(dim=1)
        return x, log_det_jacobian

    def inverse(self, z):
        for mask, st_net in reversed(list(zip(self.masks, self.s_t_layers))):
            mask = mask.to(z.device)
            z_masked = z * mask
            st = st_net(z_masked * (1 - mask))
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            z = z_masked + (1 - mask) * ((z - t) * torch.exp(-s))
        return z