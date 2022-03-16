import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from ffmlp import FFMLP

from .renderer import NeRFRenderer

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)#.clamp(0, 1000)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply # why doesn't work?


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=5,
                 hidden_dim=64,
                 cuda_ray=False,
                 ):
        super().__init__(cuda_ray)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding)

        self.sigma_net = FFMLP(
            input_dim=self.in_dim, 
            output_dim=4,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

    
    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # shift origin
        #x = x + self.origin

        # sigma
        x = self.encoder(x, size=bound)
        h = self.sigma_net(x)

        #sigma = trunc_exp(h[..., 0])
        sigma = F.softplus(h[..., 0])
        color = torch.sigmoid(h[..., 1:])
    
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        # shift origin
        #x = x + self.origin

        x = self.encoder(x, size=bound)
        h = self.sigma_net(x)

        #sigma = trunc_exp(h[..., 0])
        sigma = F.softplus(h[..., 0])
        sigma = sigma.view(*prefix)

        return sigma