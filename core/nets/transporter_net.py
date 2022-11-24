import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F

from ..datasets.train_dataset import make_3d_grid
from . import decoder
from .encoder import encoder_dict
import ipdb

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder,
    'recon_coord': decoder.Recon_Coord
}


def normalize(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return tensor / ((tensor ** 2).sum(dim, keepdim=True).sqrt() + 1.0e-5)


def skew(vector: torch.Tensor) -> torch.Tensor:
    # vector: B*3
    result = torch.zeros(vector.size(0), 3, 3).to(vector.device)
    result[:, 0, 1] = -vector[:, 2]
    result[:, 0, 2] = vector[:, 1]
    result[:, 1, 0] = vector[:, 2]
    result[:, 1, 2] = -vector[:, 0]
    result[:, 2, 0] = -vector[:, 1]
    result[:, 2, 1] = vector[:, 0]
    return result


def batch_eye(batch_size: int, dim: int) -> torch.Tensor:
    e = torch.eye(dim)
    e = e.unsqueeze(0)
    e = e.repeat(batch_size, 1, 1)
    return e


def rotation_matrix_from_axis(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    theta = theta.unsqueeze(-1).unsqueeze(-1)
    batch_size = axis.size(0)
    R = batch_eye(batch_size, 3).to(axis.device) * torch.cos(theta)
    R += skew(axis) * torch.sin(theta)
    R += (1 - torch.cos(theta)) * torch.bmm(axis.unsqueeze(2), axis.unsqueeze(1))
    return R


def angle_axis_to_rotation_matrix(angle_axis, theta):
    # Stolen from PyTorch geometry library. Modified for our code
    angle_axis_shape = angle_axis.shape
    angle_axis_ = angle_axis.contiguous().view(-1, 3)
    theta_ = theta.contiguous().view(-1, 1)

    k_one = 1.0
    normed_axes = angle_axis_ / angle_axis_.norm(dim=-1, keepdim=True)
    wx, wy, wz = torch.chunk(normed_axes, 3, dim=1)
    cos_theta = torch.cos(theta_)
    sin_theta = torch.sin(theta_)

    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(list(angle_axis_shape[:-1]) + [3, 3])


def gaussian_grid(gird_reso, mu, std=0.2):
    # features: (N, K, H, W)
    depth, height, width = gird_reso
    # mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_x, mu_y, mu_z = mu[:, :, 0:1], mu[:, :, 1:2], mu[:, :, 2:3]
    z = torch.linspace(-1.0, 1.0, depth, dtype=mu.dtype, device=mu.device)
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_z, mu_y, mu_x = mu_z.unsqueeze(-1).unsqueeze(-1), \
                        mu_y.unsqueeze(-1).unsqueeze(-1), \
                        mu_x.unsqueeze(-1).unsqueeze(-1)

    z = torch.reshape(z, [1, 1, depth, 1, 1])
    y = torch.reshape(y, [1, 1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, 1, width])
    
    g_z = torch.pow(z - mu_z, 2)
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    
    if isinstance(std, list):
        inv_std = torch.ones_like(g_z)
        for b in range(g_z.shape[0]):
            inv_std[b] /= std[b]
    else:
        inv_std = 1 / std

    dist = (g_x + g_y + g_z) * inv_std**2
    g_zyx = torch.exp(-dist)
    # g_yx = g_yx.permute([0, 2, 3, 1])
    return g_zyx


def spatial_softmax(features):
    """Compute softmax over the spatial dimensions

    Compute the softmax over heights and width

    Args
    ----
    features: tensor of shape [N, C, H, W]
    """
    features_reshape = features.reshape(features.shape[:-3] + (-1,))
    output = F.softmax(features_reshape, dim=-1)
    output = output.reshape(features.shape)
    return output


def compute_keypoint_location_mean(features):
    S_z = features.sum(-1).sum(-1)  # N, K, H
    S_y = features.sum(-1).sum(-2)  # N, K, W
    S_x = features.sum(-2).sum(-2)  # N, K, W

    # N, K
    u_z = S_z.mul(torch.linspace(-1, 1, S_z.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    u_y = S_y.mul(torch.linspace(-1, 1, S_y.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    u_x = S_x.mul(torch.linspace(-1, 1, S_x.size(-1), dtype=features.dtype, device=features.device)).sum(-1)

    return torch.stack((u_x, u_y, u_z), -1) # N, K, 2


class KeypointDecoder(nn.Module):
    ''' 
    '''

    def __init__(self, kp_num):
        super().__init__()
        self.kp_num = kp_num
        self.fc1 = nn.Linear(5000, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, kp_num*3)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        #B, 5000, 32
        
        x = torch.max(x, -1, keepdim=False)[0]

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x).view(x.shape[0], self.kp_num, 3)
        
        x = F.softplus(x)
        kps = x / (x + 1)

        return kps


class Transporter(nn.Module):
    ''' 

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, **cfg):
        super().__init__()

        encoder = cfg['encoder']
        decoder = cfg['decoder']
        dim = 3
        c_dim = cfg['c_dim']
        encoder_kwargs = cfg['encoder_kwargs']
        
        padding = cfg['padding']
        z_max = cfg['z_max']
        z_min = cfg['z_min']
        
        self.reso_grid = encoder_kwargs['grid_resolution']
        self.dim_size = self.reso_grid[0] * self.reso_grid[1] * self.reso_grid[2]
        
        self.sigmoid = cfg["sigmoid"]

        self.train_pose_only = cfg.get("train_pose_only", False)

        self.parameters_to_train = []

        # pointcloud encoder
        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
            **encoder_kwargs)
        if not self.train_pose_only:
            self.parameters_to_train += list(self.encoder.parameters())

        # implicit decoder
        decoder_occup = decoder.get('decoder_occup', None)
        if decoder_occup is not None:
            self.decoder_occup = decoder_dict[decoder_occup['decoder_type']](
                dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
                **decoder_occup['decoder_kwargs']
            )
            if not self.train_pose_only:
                self.parameters_to_train += list(self.decoder_occup.parameters())
        else:
            self.decoder_occup = None
        
        # kp_encoder
        kp_encoder = cfg['keypointnet']['encoder']
        kp_encoder_kwargs = cfg['keypointnet']['encoder_kwargs']
        self.kp_encoder = encoder_dict[kp_encoder](
            dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
            **kp_encoder_kwargs)
        if not self.train_pose_only:
            self.parameters_to_train += list(self.kp_encoder.parameters())

        # kp_decoder
        kp_decoder_kwargs = cfg['keypointnet']['decoder_kwargs']
        # self.kp_decoder = KeypointDecoder(kp_num=kp_decoder_kwargs['kp_num'])
        std = kp_decoder_kwargs['std'] #1 / self.reso_grid[0] * 3
        if std == -1:
            self.std = {'Refrigerator': 0.2, 'Microwave': 0.2, 'FoldingChair': 0.15, 'Laptop': 0.15, 
                        'Stapler': 0.15, 'TrashCan': 0.15, 'Toilet': 0.15, "StorageFurniture": 0.15, 
                        'Kettle': 0.1}
        else:
            self.std = std
        # self.parameters_to_train += list(self.kp_decoder.parameters())

        # posenet
        if cfg.get('posenet', None) is not None:
            pose_encoder = cfg['posenet']['pose_encoder']
            pose_encoder_kwargs = cfg['posenet']['pose_encoder_kwargs']
            self.pose_encoder = encoder_dict[pose_encoder](**pose_encoder_kwargs)
            self.parameters_to_train += list(self.pose_encoder.parameters())
        else:
            self.pose_encoder = None
        

    def forward(self, inputs1, inputs2, inputs3=None, p=None, obj_name='Laptop', **kwargs):

        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
        '''
        self.outputs = {}
        c12 = {}
        if self.train_pose_only:
            with torch.no_grad():
                kps1_fea = self.kp_encoder(inputs1)
                kps2_fea = self.kp_encoder(inputs2)

                kps1 = compute_keypoint_location_mean(spatial_softmax(kps1_fea['grid']))
                kps2 = compute_keypoint_location_mean(spatial_softmax(kps2_fea['grid']))

                self.outputs['kps1'] = kps1.detach()
                self.outputs['kps2'] = kps2.detach()

        else:
            kps1_fea = self.kp_encoder(inputs1)
            kps2_fea = self.kp_encoder(inputs2)

            kps1 = compute_keypoint_location_mean(spatial_softmax(kps1_fea['grid']))
            kps2 = compute_keypoint_location_mean(spatial_softmax(kps2_fea['grid']))
            
            self.outputs['kps1'] = kps1
            self.outputs['kps2'] = kps2

            if p is not None:
                c1 = self.encode_inputs(inputs1)
                c2 = self.encode_inputs(inputs2)

                if isinstance(self.std, dict):
                    std = [self.std[name] for name in obj_name]
                    kps1_heatgrid = gaussian_grid(self.reso_grid, kps1, std)
                    kps2_heatgrid = gaussian_grid(self.reso_grid, kps2, std)
                else:
                    kps1_heatgrid = gaussian_grid(self.reso_grid, kps1, self.std)
                    kps2_heatgrid = gaussian_grid(self.reso_grid, kps2, self.std)

                c12['grid'] = self.transport(kps1_heatgrid.detach(), kps2_heatgrid,
                                        c1['grid'].detach(), c2['grid'])
            
                self.decode_occup(p, c12, index='transp', **kwargs)
                self.decode_occup(p, c2, index='ori', **kwargs)

        if self.pose_encoder is not None:
            self.outputs['axis12'], self.outputs['state12'], \
                self.outputs['pred_joint12'] = self.decode_pose(inputs1, inputs2)

            if inputs_middle is not None:
                self.outputs['axis1m'], self.outputs['state1m'], _ = self.decode_pose(inputs1, inputs_middle)

                self.outputs['axis2m'], self.outputs['state2m'], _ = self.decode_pose(inputs2, inputs_middle)

        return self.outputs


    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0)

        return c


    def transport(self, source_keypoints, target_keypoints, source_features,
                    target_features):
        ''' Returns xxxxx.

        Args:
            
        '''
        # mask1 = c1.new_zeros(c1.shape[0], self.dim_size)
        # mask2 = c2.new_zeros(c2.shape[0], self.dim_size)
        
        # kps1_index = coordinate2index(kps1, self.reso_grid, coord_type='3d').squeeze(1)
        # kps2_index = coordinate2index(kps2, self.reso_grid, coord_type='3d').squeeze(1)

        # mask1.scatter_(1, kps1_index, 1)
        # mask2.scatter_(1, kps2_index, 1)

        # kps1_heatgrid = mask1.reshape(c1.shape[0], 1, self.reso_grid[0], self.reso_grid[1], self.reso_grid[2])
        # kps2_heatgrid = mask2.reshape(c1.shape[0], 1, self.reso_grid[0], self.reso_grid[1], self.reso_grid[2])

        # suppress features from image a, around both keypoint locations.
        # c12 = ((1 - kps1_heatgrid) * (1 - kps2_heatgrid) * c1)

        # # copy features from image b around keypoints for image b.
        # c12 += (kps2_heatgrid * c2)        

        """
        Args
        ====
        source_keypoints (N, K, Z, H, W)
        target_keypoints (N, K, Z, H, W)
        source_features (N, D, Z, H, W)
        target_features (N, D, Z, H, W)

        Returns
        =======
        """
        out = source_features
        for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
            out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
        return out


    def decode_pose(self, inputs1, inputs2):
        out_pose12 = self.pose_encoder(inputs1, inputs2)
        axis = out_pose12[:, :3] / (torch.norm(out_pose12[:, :3], dim=1, keepdim=True) + 1e-6)
        state = out_pose12[:, 3:4]
        # rot_mat = rotation_matrix_from_axis(axis, state)
        # translate = axis * state
        pred_joint = out_pose12[:, 4]

        return axis, state, pred_joint #, rot_mat, translate, 

    
    def decode_occup(self, p, c, index, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        
        logits = self.decoder_occup(p, c, **kwargs)
        logits = logits.squeeze(-1)

        if self.sigmoid:
            self.outputs['occ_'+index] = torch.sigmoid(logits)
        else:
            p_r = dist.Bernoulli(logits=logits)
            self.outputs['occ_'+index] = p_r


class Transporter_v2(nn.Module):
    ''' 

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, **cfg):
        super().__init__()

        encoder = cfg['encoder']
        decoder = cfg['decoder']
        dim = 3
        c_dim = cfg['c_dim']
        encoder_kwargs = cfg['encoder_kwargs']
        
        padding = cfg['padding']
        z_max = cfg['z_max']
        z_min = cfg['z_min']
        
        self.reso_grid = encoder_kwargs['grid_resolution']
        self.dim_size = self.reso_grid[0] * self.reso_grid[1] * self.reso_grid[2]
        
        self.sigmoid = cfg["sigmoid"]

        self.parameters_to_train = []

        # pointcloud encoder
        self.encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
            **encoder_kwargs)
        self.parameters_to_train += list(self.encoder.parameters())

        # implicit decoder
        decoder_occup = decoder.get('decoder_occup', None)
        if decoder_occup is not None:
            self.decoder_occup = decoder_dict[decoder_occup['decoder_type']](
                dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
                **decoder_occup['decoder_kwargs']
            )
            self.parameters_to_train += list(self.decoder_occup.parameters())
            self.decoder_occup_type = decoder_occup['decoder_type']
        else:
            self.decoder_occup = None
        
        # kp_encoder
        kp_encoder = cfg['keypointnet']['encoder']
        kp_encoder_kwargs = cfg['keypointnet']['encoder_kwargs']
        self.kp_encoder = encoder_dict[kp_encoder](
            dim=dim, c_dim=c_dim, padding=padding, z_max=z_max, z_min=z_min,
            **kp_encoder_kwargs)
        self.parameters_to_train += list(self.kp_encoder.parameters())

        # kp_decoder
        kp_decoder_kwargs = cfg['keypointnet']['decoder_kwargs']
        # self.kp_decoder = KeypointDecoder(kp_num=kp_decoder_kwargs['kp_num'])
        std = kp_decoder_kwargs['std'] #1 / self.reso_grid[0] * 3
        if std == -1:
            self.std = {'Refrigerator': 0.2, 'Microwave': 0.2, 'FoldingChair': 0.15, 'Laptop': 0.15, 
                        'Stapler': 0.15, 'TrashCan': 0.15, 'Toilet': 0.15, "StorageFurniture": 0.15, 
                        'Kettle': 0.1}
        else:
            self.std = std
        

    def forward(self, inputs1, inputs2, inputs3=None, p=None, obj_name='Laptop', **kwargs):

        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
        '''
        self.outputs = {}
        c12 = {}
        
        kps1_fea, kps2_fea = self.kp_encoder(inputs1, inputs2)
        kps1 = compute_keypoint_location_mean(spatial_softmax(kps1_fea['grid']))
        kps2 = compute_keypoint_location_mean(spatial_softmax(kps2_fea['grid']))
        
        self.outputs['kps1'] = kps1
        self.outputs['kps2'] = kps2

        if inputs3 is not None:
            kps2_fea_, kps3_fea = self.kp_encoder(inputs2, inputs3)
            kps2_ = compute_keypoint_location_mean(spatial_softmax(kps2_fea_['grid']))
            kps3 = compute_keypoint_location_mean(spatial_softmax(kps3_fea['grid']))
            self.outputs['kps2_'] = kps2_
            self.outputs['kps3'] = kps3

        if p is not None:
            c1 = self.encode_inputs(inputs1)
            c2 = self.encode_inputs(inputs2)

            if isinstance(self.std, dict):
                std = [self.std[name] for name in obj_name]
                kps1_heatgrid = gaussian_grid(self.reso_grid, kps1, std)
                kps2_heatgrid = gaussian_grid(self.reso_grid, kps2, std)
            else:
                kps1_heatgrid = gaussian_grid(self.reso_grid, kps1, self.std)
                kps2_heatgrid = gaussian_grid(self.reso_grid, kps2, self.std)

            c12['grid'] = self.transport(kps1_heatgrid.detach(), kps2_heatgrid,
                                    c1['grid'].detach(), c2['grid'])

            if self.decoder_occup_type == 'recon_coord':
                self.decode_coord(c12, index='transp', **kwargs)
                self.decode_coord(c2, index='ori', **kwargs)
            else: 
                self.decode_occup(p, c12, index='transp', **kwargs)
                self.decode_occup(p, c2, index='ori', **kwargs)

        return self.outputs


    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0)

        return c


    def transport(self, source_keypoints, target_keypoints, source_features,
                    target_features):
        
        """
        Args
        ====
        source_keypoints (N, K, Z, H, W)
        target_keypoints (N, K, Z, H, W)
        source_features (N, D, Z, H, W)
        target_features (N, D, Z, H, W)

        Returns
        =======
        """
        out = source_features
        for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
            out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
        return out

    
    def decode_occup(self, p, c, index, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        
        logits = self.decoder_occup(p, c, **kwargs)
        logits = logits.squeeze(-1)

        if self.sigmoid:
            self.outputs['occ_'+index] = torch.sigmoid(logits)
        else:
            p_r = dist.Bernoulli(logits=logits)
            self.outputs['occ_'+index] = p_r


    def decode_coord(self, c, index, **kwargs):

        p = make_3d_grid([-0.5, 0.5, 64],
                        [-0.5, 0.5, 64],
                        [-0.5, 0.5, 64],
                        type='HWD')
        
        p = p.unsqueeze(0).repeat(c.shape[0]).cuda()
        coords = self.decoder_occup(p, c, **kwargs)
        
        self.outputs['coords_'+index] = coords



