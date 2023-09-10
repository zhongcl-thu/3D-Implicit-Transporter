import torch
import torch.nn as nn
from torch import distributions as dist
import torch.nn.functional as F

from . import decoder
from .encoder import encoder_dict

import ipdb

# Decoder dictionary
decoder_dict = {
    'simple_local': decoder.LocalDecoder,
    'simple_local_crop': decoder.PatchLocalDecoder,
    'simple_local_point': decoder.LocalPointDecoder,
}


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
        dim = cfg.get('dim', 3)
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
            dim=dim, padding=padding, z_max=z_max, z_min=z_min,
            **encoder_kwargs)
        self.parameters_to_train += list(self.encoder.parameters())

        # implicit decoder
        decoder_occup = decoder.get('decoder_occup', None)
        if decoder_occup is not None:
            self.decoder_occup = decoder_dict[decoder_occup['decoder_type']](
                dim=dim, c_dim=encoder_kwargs['c_dim'], padding=padding, z_max=z_max, z_min=z_min,
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
            dim=dim, padding=padding, z_max=z_max, z_min=z_min,
            **kp_encoder_kwargs)
        self.parameters_to_train += list(self.kp_encoder.parameters())

        # kp_decoder
        kp_decoder_kwargs = cfg['keypointnet']['decoder_kwargs']
        std = kp_decoder_kwargs['std'] #1 / self.reso_grid[0] * 3
        if std == -1:
            self.std = {'Refrigerator': 0.2, 'Microwave': 0.2, 'FoldingChair': 0.15, 'Laptop': 0.15, 
                        'Stapler': 0.15, 'TrashCan': 0.15, 'Toilet': 0.15, "StorageFurniture": 0.15, 
                        'Kettle': 0.1}
        else:
            self.std = std
        self.nsample = kp_decoder_kwargs.get('nsample', None)
        

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




