import torch
import torch.nn as nn
from core.utils.procrustes import rigid_transform_3d, quaternion_to_axis_angle
from pytorch3d.transforms import matrix_to_quaternion

import ipdb

def cosine(pred_axis, gt_axis, ambiguity=False):
    # pred: B * N * 3
    # target: B * 3
    if ambiguity:
        cosine_sim_0 = torch.einsum("bnm,bm->bn", pred_axis, -gt_axis)
        cosine_sim_1 = torch.einsum("bnm,bm->bn", pred_axis, gt_axis)
        cosine_sim_max = torch.maximum(cosine_sim_0, cosine_sim_1)
    else:
        cosine_sim_max = torch.einsum("bnm,bm->bn", pred_axis, gt_axis)
    return cosine_sim_max


class Kp_consist_loss (nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'kp_cons_loss'
        self.weight = weight

    def forward(self, kps2_, kps2, **kw):
        B, N, _ =  kps2.shape
        # kps2 loss
        kp_consist_loss = torch.abs(kps2 - kps2_).sum() / B
        
        return kp_consist_loss


class Pose_loss (nn.Module):
    def __init__(self, weight, corr_weight=1.0, axis_weight=1.0, **kw):
        nn.Module.__init__(self)
        self.name = 'pose_loss'
        self.weight = weight
        self.corr_weight = corr_weight
        self.axis_weight = axis_weight


    def pose_to_axis_angle(self, T_21):
        axis_angle_21 = quaternion_to_axis_angle(matrix_to_quaternion(T_21[:, :3, :3]))
        angles_21 = torch.norm(axis_angle_21, dim=1, keepdim=True) # B, 1
        axis_21 = axis_angle_21 / (angles_21 + 1e-6) # B, 3

        return angles_21, axis_21


    def translation_norm(self, T_21):
        T_21_norm = torch.norm(T_21[:, :3, 3], dim=1, keepdim=True) # B, 1
        return T_21[:, :3, 3] / T_21_norm
    

    def corr_loss(self, T_21, kps1, kps2):
        kps2_pred = (T_21[:, :3, :3] @ kps1.transpose(2, 1) + T_21[:, :3, 3:]).transpose(2, 1) #B, N, 3
        kp_corr_loss1 = torch.abs(kps2 - kps2_pred).sum()  # 1

        return kp_corr_loss1


    def forward(self, kps1, kps2, kps2_, kps3, **kw):
        B, N, _ = kps1.shape
        
        T_21, valid_svd21 = rigid_transform_3d(kps1, kps2) # SVD
        T_32, valid_svd32 = rigid_transform_3d(kps2_, kps3) # SVD

        _, axis_21 = self.pose_to_axis_angle(T_21)
        _, axis_32 = self.pose_to_axis_angle(T_32)

        axis_sim = (axis_21 * axis_32).sum(1) # B
        axis_loss = torch.minimum(1 - axis_sim, axis_sim + 1) # B
        if torch.isnan(axis_loss.mean()):
            print(T_21)
            print(T_32)
            ipdb.set_trace()

        kp_corr_loss1 = self.corr_loss(T_21, kps1, kps2) / B
        kp_corr_loss2 = self.corr_loss(T_32, kps2_, kps3) / B
        kp_corr_loss = (kp_corr_loss1  + kp_corr_loss2) / 2

        if torch.isnan(kp_corr_loss):
            print(T_21)
            print(T_32)
            ipdb.set_trace()

        return axis_loss.mean() * self.axis_weight + kp_corr_loss * self.corr_weight

