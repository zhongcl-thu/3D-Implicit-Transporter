import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion


def integrate_trans(R, t):
    """
    Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
    Input
        - R: [3, 3] or [bs, 3, 3], rotation matrix
        - t: [3, 1] or [bs, 3, 1], translation matrix
    Output
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    """
    if len(R.shape) == 3:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4)[None].repeat(R.shape[0], 1, 1).to(R.device)
        else:
            trans = np.eye(4)[None]
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t.view([-1, 3, 1])
    else:
        if isinstance(R, torch.Tensor):
            trans = torch.eye(4).to(R.device)
        else:
            trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3:4] = t
    return trans


def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    try:
        U, S, Vt = torch.svd(H.cpu())
        U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
        delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
        eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
        eye[:, -1, -1] = delta_UV
        R = Vt @ eye @ U.permute(0, 2, 1)
        t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
        # warp_A = transform(A, integrate_trans(R,t))
        # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
        return integrate_trans(R, t), True
    except:
        return torch.eye(4).unsqueeze(0).repeat(A.shape[0]).to(weights.device), False


def pose_to_axis_angle(T_21):
    axis_angle_21 = quaternion_to_axis_angle(matrix_to_quaternion(T_21[:, :3, :3]))
    angles_21 = torch.norm(axis_angle_21, dim=1, keepdim=True) # B, 1
    axis_21 = axis_angle_21 / angles_21 # B, 3

    return angles_21, axis_21

# def rigid_transform_3d(A, B, weights=None, weight_threshold=0):
#     """ 
#     Input:
#         - A:       [bs, num_corr, 3], source point cloud
#         - B:       [bs, num_corr, 3], target point cloud
#         - weights: [bs, num_corr]     weight for each correspondence 
#         - weight_threshold: float,    clips points with weight below threshold
#     Output:
#         - R, t 
#     """
#     bs = A.shape[0]
#     if weights is None:
#         weights = torch.ones_like(A[:, :, 0])
#     weights[weights < weight_threshold] = 0
#     # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

#     # find mean of point cloud
#     centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
#     centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

#     # subtract mean
#     Am = A #- centroid_A
#     Bm = B #- centroid_B

#     # construct weight covariance matrix
#     Weight = torch.diag_embed(weights)
#     H = Am.permute(0, 2, 1) @ Weight @ Bm

#     # find rotation
#     U, S, Vt = torch.svd(H.cpu())
#     U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
#     delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
#     eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
#     eye[:, -1, -1] = delta_UV
#     R = Vt @ eye @ U.permute(0, 2, 1)
#     t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
#     # warp_A = transform(A, integrate_trans(R,t))
#     # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
#     return integrate_trans(R, t)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles