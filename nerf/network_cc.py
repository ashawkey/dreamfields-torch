import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from nerf.renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 resolution=[256] * 3,
                 rank_line=32,
                 rank_plane=32,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        self.resolution = np.asarray(resolution)

        self.out_dim = 4

        self.vec_ids = [0, 1, 2]
        self.mat_ids = [[0, 1], [0, 2], [1, 2]]

        self.rank_line = rank_line
        self.rank_plane = rank_plane

        # line
        self.U = nn.ParameterList()
        for i in range(len(self.vec_ids)):
            vec_id = self.vec_ids[i]
            self.U.append(nn.Parameter(torch.randn(1, rank_line, resolution[vec_id], 1) * 0.1)) # [1, R, H, 1]

        # plane
        self.V = nn.ParameterList()
        for i in range(len(self.mat_ids)):
            mat_id_0, mat_id_1 = self.mat_ids[i]
            self.V.append(nn.Parameter(torch.randn(1, rank_plane, resolution[mat_id_1], resolution[mat_id_0]) * 0.1)) # [1, R, H, W]

        # singular values (for line and plane, separately)
        self.S = nn.ParameterList()
        self.S.append(nn.Parameter(torch.ones(self.out_dim, rank_line)))
        self.S.append(nn.Parameter(torch.ones(self.out_dim, rank_plane)))
        
        torch.nn.init.kaiming_normal_(self.S[0].data)
        torch.nn.init.kaiming_normal_(self.S[1].data)


    def transform(self, x):
        # x: [N, 3], in [-bound, bound]
        # y: transformed x in oid's coordinate system, and normalized into [-1, 1]

        #x = x + self.origin
    
        aabb = self.aabb_train if self.training else self.aabb_infer
        y = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # in [-1, 1] (may have outliers, but no matter since grid_sample use zero padding.)
    
        return y

    
    def get_feat(self, x):
        # x: [N, 3], in [-1, 1]

        N = x.shape[0]

        vec_coord = torch.stack((x[..., self.vec_ids[0]], x[..., self.vec_ids[1]], x[..., self.vec_ids[2]]))
        vec_coord = torch.stack((torch.zeros_like(vec_coord), vec_coord), dim=-1).detach().view(3, -1, 1, 2) # [3, N, 1, 2], fake 2d coord

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).detach().view(3, -1, 1, 2) # [3, N, 1, 2]
    

        vec_feat = F.grid_sample(self.U[0], vec_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.U[1], vec_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.U[2], vec_coord[[2]], align_corners=True).view(-1, N) # [R1, N]

        mat_feat = F.grid_sample(self.V[0], mat_coord[[0]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.V[1], mat_coord[[1]], align_corners=True).view(-1, N) * \
                   F.grid_sample(self.V[2], mat_coord[[2]], align_corners=True).view(-1, N) # [R2, N]
        
        S_vec = self.S[0]
        S_mat = self.S[1]

        vec_feat = S_vec @ vec_feat # [out_dim, N]
        mat_feat = S_mat @ mat_feat # [out_dim, N]

        hybrid_feat = (vec_feat + mat_feat).T.contiguous() # [out_dim, N] --> [N, out_dim]

        return hybrid_feat

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        # normalize to [-1, 1]
        x_model = self.transform(x)

        feat = self.get_feat(x_model) # [N, out_dim]
        sigma = trunc_exp(feat[..., 0])

        return {
            'sigma': sigma,
            'feat': feat,
        }

    # allow masked inference
    def color(self, x, d, mask=None, feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        # feat: [N, out_dim]
        N = x.shape[0]

        h = feat[..., 1:] # [N, 3]
        rgbs = torch.sigmoid(h)

        return rgbs

    # L1 penalty for loss
    def density_loss(self):
        loss = 0
        for i in range(len(self.U)):
            loss += torch.mean(torch.abs(self.U[i])) + torch.mean(torch.abs(self.V[i]))
        return loss
    

    @torch.no_grad()
    def upsample_model(self, resolution):
        for i in range(len(self.U)):
            vec_id = self.vec_ids[i % 3]
            self.U[i] = torch.nn.Parameter(F.interpolate(self.U[i].data, size=(resolution[vec_id], 1), mode='bilinear', align_corners=True))
            
        for i in range(len(self.V)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.V[i] = torch.nn.Parameter(F.interpolate(self.V[i].data, size=(resolution[mat_id_1], resolution[mat_id_0]), mode='bilinear', align_corners=True))
            
        self.resolution = resolution


    @torch.no_grad()
    def shrink_model(self):
        # shrink aabb_train and the model so it only represents the space inside aabb_train.

        H = self.density_grid.shape[1]
        half_grid_size = self.bound / H
        thresh = min(0.01, self.mean_density)

        # get new aabb from the coarsest density grid (TODO: from the finest that covers current aabb?)
        valid_grid = self.density_grid[self.cascade - 1] > thresh # [H, W, D]
        valid_pos = torch.nonzero(valid_grid) # [Nz, 3], in [0, H - 1]
        #plot_pointcloud(valid_pos.detach().cpu().numpy()) # lots of noisy outliers in hashnerf...
        valid_pos = (2 * valid_pos / (H - 1) - 1) * (self.bound - half_grid_size) # [Nz, 3], in [-b+hgs, b-hgs]
        min_pos = valid_pos.amin(0) - half_grid_size # [3]
        max_pos = valid_pos.amax(0) + half_grid_size # [3]

        # shrink model
        reso = torch.LongTensor(self.resolution).to(self.aabb_train.device)
        units = (self.aabb_train[3:] - self.aabb_train[:3]) / reso
        tl = (min_pos - self.aabb_train[:3]) / units
        br = (max_pos - self.aabb_train[:3]) / units
        tl = torch.round(tl).long().clamp(min=0)
        br = torch.minimum(torch.round(br).long(), reso)
        
        for i in range(len(self.U)):
            vec_id = self.vec_ids[i % 3]
            self.U[i] = nn.Parameter(self.U[i].data[..., tl[vec_id]:br[vec_id], :])

        for i in range(len(self.V)):
            mat_id_0, mat_id_1 = self.mat_ids[i % 3]
            self.V[i] = nn.Parameter(self.V[i].data[..., tl[mat_id_1]:br[mat_id_1], tl[mat_id_0]:br[mat_id_0]])
        
        self.aabb_train = torch.cat([min_pos, max_pos], dim=0) # [6]

        print(f'[INFO] shrink slice: {tl.cpu().numpy().tolist()} - {br.cpu().numpy().tolist()}')
        print(f'[INFO] new aabb: {self.aabb_train.cpu().numpy().tolist()}')


    # optimizer utils
    def get_params(self, lr1, lr2=None):
        if lr2 is None:
            lr2 = lr1
        return [
            {'params': self.U, 'lr': lr1},
            {'params': self.V, 'lr': lr1},
            {'params': self.S, 'lr': lr2},
        ]