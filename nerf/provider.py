import torch
import numpytorch as np
from torch.utils.data import Dataset

import trimesh

def normalize(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-10)

"""
	const float axis_size = 0.025f;
	const Vector3f *xforms = (const Vector3f*)&xform;
	Vector3f pos = xforms[3];
	add_debug_line(world2proj, list, pos, pos+axis_size*xforms[0], 0xff4040ff);
	add_debug_line(world2proj, list, pos, pos+axis_size*xforms[1], 0xff40ff40);
	add_debug_line(world2proj, list, pos, pos+axis_size*xforms[2], 0xffff4040);
	float xs=axis_size*aspect;
	float ys=axis_size;
	float zs=axis_size*2.f*aspect;
	Vector3f a = pos + xs * xforms[0] + ys * xforms[1] + zs * xforms[2];
	Vector3f b = pos - xs * xforms[0] + ys * xforms[1] + zs * xforms[2];
	Vector3f c = pos - xs * xforms[0] - ys * xforms[1] + zs * xforms[2];
	Vector3f d = pos + xs * xforms[0] - ys * xforms[1] + zs * xforms[2];
	add_debug_line(world2proj, list, pos, a, col);
	add_debug_line(world2proj, list, pos, b, col);
	add_debug_line(world2proj, list, pos, c, col);
	add_debug_line(world2proj, list, pos, d, col);
	add_debug_line(world2proj, list, a, b, col);
	add_debug_line(world2proj, list, b, c, col);
	add_debug_line(world2proj, list, c, d, col);
	add_debug_line(world2proj, list, d, a, col);
"""


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    objects = [axes]

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


class NeRFDataset(Dataset):
    def __init__(self, type='train', H=800, W=800, radius=2, fovy=90, size=1000):
        super().__init__()

        self.type = type # train, val, test
        self.size = size
        self.radius = radius

        self.H = H
        self.W = W

        # TODO: for type = test or val, should fix the sampled cameras? (e.g. 360 deg rot)

        # intrinsics
        self.fovy = np.radians(fovy) 

        fl_y = self.H / (2 * np.tan(self.fovy / 2))
        fl_x = fl_y
        cx = self.H / 2
        cy = self.W / 2

        self.intrinsic = np.eye(3, dtype=np.float32)
        self.intrinsic[0, 0] = fl_x
        self.intrinsic[1, 1] = fl_y
        self.intrinsic[0, 2] = cx
        self.intrinsic[1, 2] = cy

        # preload
        self.intrinsic = torch.from_numpy(self.intrinsic).cuda()
        self.generate_poses()

    def __len__(self):
        return self.size

    def generate_poses(self):
        # generate random poses in batch

        #thetas = np.rand(self.size) * np.pi
        thetas = np.rand(self.size) * np.pi / 3 + np.pi / 3 # limit elevation
        phis = np.rand(self.size) * 2 * np.pi

        centers = np.stack([
            self.radius * np.sin(thetas) * np.sin(phis),
            self.radius * np.cos(thetas),
            self.radius * np.sin(thetas) * np.cos(phis),
        ], axis=-1) # [B, 3]

        forward_vector = - normalize(centers) # camera direction (OpenGL convention!)
        up_vector = np.array([0, 1, 0], dtype=np.float32).unsqueeze(0).torch_repeat(self.size, 1)
        right_vector = normalize(np.cross(forward_vector, up_vector, axis=-1))
        up_vector = normalize(np.cross(right_vector, forward_vector, axis=-1))

        poses = np.eye(4, dtype=np.float32).unsqueeze(0).torch_repeat(self.size, 1, 1)
        poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
        poses[:, :3, 3] = centers

        #visualize_poses(poses)

        self.poses = torch.from_numpy(poses).cuda()

        # TODO: classify thetas/phis to explicit directions (front, side, back, top, bottom)
        self.dirs = get_view_direction(thetas, phis)



    def __getitem__(self, index):
            
        results = {
            'pose': self.poses[index],
            'intrinsic': self.intrinsic,
            'index': index,
            'H': str(self.H),
            'W': str(self.W),
            'dir': self.dirs[index],
        }
                
        return results