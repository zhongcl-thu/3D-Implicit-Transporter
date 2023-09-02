from typing import List
import torch
import torch.nn as nn
import math
import numpy as np


# code from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def mlp(input_dim, layer_dims, bn=None):
    layer_dims.insert(0, input_dim)
    layers = nn.ModuleList()
    for i in range(len(layer_dims) - 2):
        layers.append(
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.BatchNorm1d(layer_dims[i + 1]) if bn else nn.Identity(),
                nn.ReLU()
            )
        )
    
    layers.append(
        nn.Linear(layer_dims[-2], layer_dims[-1])
    )
    return nn.Sequential(*layers)


def mlp_conv(input_dim, layer_dims, bn=None):
    layer_dims.insert(0, input_dim)
    layers = nn.ModuleList()
    for i in range(len(layer_dims) - 2):
        layers.append(
            nn.Sequential(
                nn.Conv1d(layer_dims[i], layer_dims[i + 1], kernel_size=1),
                nn.BatchNorm1d(layer_dims[i + 1]) if bn else nn.Identity(),
                nn.ReLU()
            )
        )
    
    layers.append(
        nn.Conv1d(layer_dims[-2], layer_dims[-1], kernel_size=1)
    )
    return nn.Sequential(*layers)


# Number of children per tree levels for 2048 output points
def get_arch(nlevels, npts):
    tree_arch = {}
    tree_arch[2] = [32, 64]
    tree_arch[4] = [4, 8, 8, 8]
    tree_arch[6] = [2, 4, 4, 4, 4, 4]
    tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch

def rifeat(points_r, points_s):
    """generate rotation invariant features
    Args:
        points_r (B x N x K x 3): 
        points_s (B x N x 1 x 3): 
    """

    # [*, 3] -> [*, 6] with compatible intra-shapes
    if points_r.shape[1] != points_s.shape[1]:
        points_r = points_r.expand(-1, points_s.shape[1], -1, -1)
    
    r_mean = torch.mean(points_r, -2, keepdim=True)
    l1, l2, l3 = r_mean - points_r, points_r - points_s, points_s - r_mean
    l1_norm = torch.norm(l1, 'fro', -1, True)
    l2_norm = torch.norm(l2, 'fro', -1, True)
    l3_norm = torch.norm(l3, 'fro', -1, True).expand_as(l2_norm)
    theta1 = (l1 * l2).sum(-1, keepdim=True) / (l1_norm * l2_norm + 1e-7)
    theta2 = (l2 * l3).sum(-1, keepdim=True) / (l2_norm * l3_norm + 1e-7)
    theta3 = (l3 * l1).sum(-1, keepdim=True) / (l3_norm * l1_norm + 1e-7)
    
    return torch.cat([l1_norm, l2_norm, l3_norm, theta1, theta2, theta3], dim=-1)


def conv_kernel(iunit, ounit, *hunits):
    layers = []
    for unit in hunits:
        layers.append(nn.Linear(iunit, unit))
        layers.append(nn.LayerNorm(unit))
        layers.append(nn.ReLU())
        iunit = unit
    layers.append(nn.Linear(iunit, ounit))
    return nn.Sequential(*layers)


class GlobalInfoProp(nn.Module):
    def __init__(self, n_in, n_global):
        super().__init__()
        self.linear = nn.Linear(n_in, n_global)

    def forward(self, feat):
        # [b, k, n_in] -> [b, k, n_in + n_global]
        tran = self.linear(feat)
        glob = tran.max(-2, keepdim=True)[0].expand(*feat.shape[:-1], tran.shape[-1])
        return torch.cat([feat, glob], -1)

        
class SparseSO3Conv(nn.Module):
    def __init__(self, rank, n_out, *kernel_interns, layer_norm=True):
        super().__init__()
        self.kernel = conv_kernel(6, rank, *kernel_interns)
        self.outnet = nn.Linear(rank, n_out)
        self.rank = rank
        self.layer_norm = nn.LayerNorm(n_out) if layer_norm else None

    def do_conv_ranked(self, r_inv_s, feat):
        # [b, n, k, rank], [b, n, k, cin] -> [b, n, cout]
        kern = self.kernel(r_inv_s).reshape(*feat.shape[:-1], self.rank)
        # PointNet max operation
        contracted = kern.max(-2)[0]
        return self.outnet(contracted)

    # pts: B x N x 3
    def forward(self, pts, nbr_pts):
        r_inv_s = rifeat(nbr_pts, torch.unsqueeze(pts, -2))
        conv = self.do_conv_ranked(r_inv_s)
        if self.layer_norm is not None:
            return self.layer_norm(conv)
        return conv