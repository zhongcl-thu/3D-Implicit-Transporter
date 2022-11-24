"""
From the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import torch.nn as nn

from .modules.Transformer import Attn, TransformerAttn
from .pointnetpp_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    sample_and_group_all
)
import ipdb

# Concat attn with original feature
# return score and abstract point index
# different decoder for occ and articulation
class PointNetPlusPlusAttnFusion(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None, out_dim=5):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        # self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        # self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        # self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        # self.fp1_corr = PointNetFeaturePropagation(
        #     in_channel=256, mlp=[256, 128, c_dim]
        # )

        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, out_dim)

        attn_type = attn_kwargs.get("type", "Transformer")
        if attn_type == "simple":
            self.attn = Attn(attn_kwargs)
        elif attn_type == "Transformer":
            self.attn = TransformerAttn(attn_kwargs)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True) # 
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        B, _, _ = xyz.shape
        xyz = xyz.permute(0, 2, 1) # B, 3, N
        l2_points_xyz2, l2_xyz2, fps_idx2 = self.encode_deep_feature(
            xyz2, return_xyz=True
        ) # l2_points_xyz2 B, 256, 128
        
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        '''encode deep feature for xyz'''
        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True) 
        # l1_xyz [B,3,512] l1_points [B,128,512]
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        # l2_xyz [B,3,128] l2_points [B,256,128]
        
        # fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        
        '''attention'''
        attn, score = self.attn(l2_points, l2_points_xyz2, True)# attn B,256,128  
        # l2_points = torch.cat((l2_points, attn), dim=1)# l2_points B,512,128  
        
        new_attn = torch.max(attn, 2)[0]
        return self.fc1(new_attn)
        

        # l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)

        # l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)
        # if return_score:
        #     return (
        #         xyz.permute(0, 2, 1),
        #         l0_points.permute(0, 2, 1),
        #         l0_points_corr.permute(0, 2, 1),
        #         fps_idx,
        #         fps_idx2,
        #         score,
        #     )
        # else:
        #     return (
        #         xyz.permute(0, 2, 1),
        #         l0_points.permute(0, 2, 1),
        #         l0_points_corr.permute(0, 2, 1),
        #     )


# Concat attn with original feature
# return score and abstract point index
# different decoder for occ and articulation
class PointNetPlusPlusAttnFusion_cross(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        # self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        # self.fp1_corr = PointNetFeaturePropagation(
        #     in_channel=256, mlp=[256, 128, c_dim]
        # )
        
        attn_type = attn_kwargs.get("type", "Transformer")
        if attn_type == "simple":
            self.attn = Attn(attn_kwargs)
        elif attn_type == "Transformer":
            self.attn = TransformerAttn(attn_kwargs)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, l1_points, l1_xyz, l0_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """
        
        l2_points, l2_xyz, l1_points, l1_xyz, l0_xyz, _ = self.encode_deep_feature(
            xyz, return_xyz=True
        ) # view 1
        
        l2_points_xyz2, l2_xyz2, l1_points_xyz2, l1_xyz2, l0_xyz2, _ = self.encode_deep_feature(
            xyz2, return_xyz=True
        ) # view 2

        attn12, score12 = self.attn(l2_points, l2_points_xyz2, True)
        attn21, score21 = self.attn(l2_points_xyz2, l2_points, True)

        l2_points = torch.cat((l2_points, attn12), dim=1)
        l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)

        l2_points_xyz2 = torch.cat((l2_points_xyz2, attn21), dim=1)
        l1_points_xyz2_back = self.fp2(l1_xyz2, l2_xyz2, l1_points_xyz2, l2_points_xyz2)
        l0_points_xyz2 = self.fp1(l0_xyz2, l1_xyz2, None, l1_points_xyz2_back)

        return (
            l0_points.permute(0, 2, 1),
            l0_points_xyz2.permute(0, 2, 1)
        )
