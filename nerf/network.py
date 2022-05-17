import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="frequency",
                 num_layers=6,
                 hidden_dim=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 4
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

    
    def forward(self, x, d):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        # shift origin
        x = x + self.origin

        # sigma
        x = self.encoder(x, size=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.softplus(h[..., 0])
        color = torch.sigmoid(h[..., 1:])

        return sigma, color

    def density(self, x):
        # x: [B, N, 3], in [-bound, bound]

        # shift origin
        x = x + self.origin

        x = self.encoder(x, size=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.softplus(h[..., 0])

        return {
            'sigma': sigma,
        }


    # optimizer utils
    def get_params(self, lr1):
        return [
            {'params': self.sigma_net.parameters(), 'lr': lr1},
        ]        