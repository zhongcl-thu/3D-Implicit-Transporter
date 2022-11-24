import torch
import torch.nn as nn
import ipdb
from ..nets.transporter_net import rotation_matrix_from_axis, angle_axis_to_rotation_matrix
from core.utils.procrustes import rigid_transform_3d, quaternion_to_axis_angle
from pytorch3d.transforms import matrix_to_quaternion


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
        axis_21 = axis_angle_21 / angles_21 # B, 3

        return angles_21, axis_21
    

    def translation_norm(self, T_21):
        T_21_norm = torch.norm(T_21[:, :3, 3], dim=1, keepdim=True) # B, 1
        return T_21[:, :3, 3] / T_21_norm
    

    def corr_loss(self, T_21, kps1, kps2):
        kps2_pred = (T_21[:, :3, :3] @ kps1.transpose(2, 1) + T_21[:, :3, 3:]).transpose(2, 1) #B, N, 3
        kp_corr_loss1 = torch.abs(kps2 - kps2_pred).sum()  # 1

        return kp_corr_loss1


    def forward(self, kps1, kps2, kps2_=None, kps3=None, gt_joint=None, 
                    pose21=None, pose32=None, scale=None, center=None, **kw):
        
        B, N, _ = kps1.shape

        if pose21 is None:
            T_21, valid_svd21 = rigid_transform_3d(kps1, kps2) # SVD
            if kps2_ is not None:
                T_32, valid_svd32 = rigid_transform_3d(kps2_, kps3)

            # if not valid_svd21 or not valid_svd32:
            #     return torch.tensor(10.0).to(kps1.device)
            angles_21, axis_21 = self.pose_to_axis_angle(T_21)
            if kps2_ is not None:
                angles_32, axis_32 = self.pose_to_axis_angle(T_32)

                axis_sim = (axis_21 * axis_32).sum(1) # B
                axis_loss = torch.minimum(1 - axis_sim, axis_sim + 1) # B
            # trans_rot_loss = torch.abs(angles_21 + angles_32).sum(1) / 2 # B

            # trans_norm_21 = self.translation_norm(T_21)
            # trans_norm_32 = self.translation_norm(T_32)
            # trans_dir_sim = (trans_norm_21 * trans_norm_32).sum(1) # B
            # trans_dir_loss = torch.minimum(1 - trans_dir_sim, trans_dir_sim + 1) # B

                batch_index_R = torch.where(gt_joint>=1)[0]
                batch_index_T = torch.where(gt_joint<0)[0]
                # batch_index_R = torch.where(gt_joint==1)[0]
                # batch_index_T = torch.where(gt_joint==0)[0]
                
                if batch_index_R.shape[0] > 0:
                    axis_loss = axis_loss[batch_index_R].mean()
                else:
                    axis_loss = 0.0

                if batch_index_T.shape[0] > 0:
                    trans_rot_loss = trans_rot_loss[batch_index_T].mean()
                    trans_dir_loss = trans_dir_loss[batch_index_T].mean()
                else:
                    trans_rot_loss = 0.0
                    trans_dir_loss = 0.0
            else:
                axis_loss = 0.0
                trans_rot_loss = 0.0
                trans_dir_loss = 0.0

        else:
            T_21 = pose21
            T_32 = pose32

            axis_loss = 0.0
            trans_loss = 0.0
            
            kps1 = kps1 * scale.unsqueeze(2) / 2 + center.unsqueeze(1) 
            kps2 = kps2 * scale.unsqueeze(2) / 2 + center.unsqueeze(1) 
            kps2_ = kps2_ * scale.unsqueeze(2) / 2 + center.unsqueeze(1) 
            kps3 = kps3 * scale.unsqueeze(2) / 2 + center.unsqueeze(1) 

        kp_corr_loss1 = self.corr_loss(T_21, kps1, kps2) / B
        if kps2_ is not None:
            kp_corr_loss2 = self.corr_loss(T_32, kps2_, kps3) / B
            kp_corr_loss = (kp_corr_loss1  + kp_corr_loss2) / 2
        else:
            kp_corr_loss = kp_corr_loss1

        return (axis_loss + trans_rot_loss + trans_dir_loss) * self.axis_weight \
                + kp_corr_loss * self.corr_weight


class Joint_cls_loss (nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'joint_cls_loss'
        self.weight = weight

        self.cri_cls = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_joint12, gt_joint, **kw):
        loss = self.cri_cls(pred_joint12.unsqueeze(1), gt_joint)

        return loss
