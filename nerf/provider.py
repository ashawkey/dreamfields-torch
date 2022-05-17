import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()

def get_view_direction(thetas, phis):
    #                   phis [B,]; thetas: [B,]
    # front = 0         0-90            
    # side (left) = 1   90-180
    # back = 2          180-270
    # side (right) = 3  270-360
    # top = 4                        0-45
    # bottom = 5                     135-180
    res = np.zeros(phis.shape[0], dtype=np.int64)
    # first determine by phis
    res[phis < (np.pi / 2)] = 0
    res[(phis >= (np.pi / 2)) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (3 * np.pi / 2))] = 2
    res[(phis >= (3 * np.pi / 2)) & (phis < (2 * np.pi))] = 3
    # override by thetas
    res[thetas < (np.pi / 4)] = 4
    res[thetas > (3 * np.pi / 4)] = 5
    return res


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', H=128, W=128, radius=3, fovy=90, size=100):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test

        self.H = H
        self.W = W
        self.radius = radius
        self.fovy = fovy
        self.size = size

        self.training = self.type in ['train', 'all']
        self.num_rays = self.opt.num_rays if self.training else -1

        fl_y = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        fl_x = fl_y
        cx = self.H / 2
        cy = self.W / 2
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # [debug] visualize poses
        #poses = rand_poses(100, 'cpu', radius=self.radius).detach().numpy()
        #visualize_poses(poses)


    def collate(self, index):

        B = len(index) # always 1

        # random pose
        poses = rand_poses(B, self.device, radius=self.radius)

        # sample a low-resolution but full image for CLIP
        rays = get_rays(poses, self.intrinsics, self.H, self.W, -1)

        return {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],    
        }

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader