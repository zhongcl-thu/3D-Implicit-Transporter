import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder.pointnet import ResnetBlockFC
from .common import normalize_coordinate, normalize_3d_coordinate, map2local
from .model_utils import get_arch, mlp, mlp_conv
import ipdb

from .encoder.pointnetpp_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    sample_and_group_all
)


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, out_dim=1,
                 hidden_size=256, n_blocks=5, leaky=False, 
                 sample_mode='bilinear', padding=0.1, z_max=0.5, z_min=-0.5,
                 desc_type='field'): #desc_type: [field, occp]
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
        self.z_max = z_max
        self.z_min = z_min

        self.desc_type = desc_type
    

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c


    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding,
                                        z_max=self.z_max, z_min=self.z_min) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, return_desc=False, **kwargs):
        
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)
        desc = []

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
            if return_desc:
                if self.desc_type == 'occp':
                    desc.append(net)

        out = self.fc_out(self.actvn(net))
        # ipdb.set_trace()
        # out = out.squeeze(-1)
        if return_desc:
            #return out, c
            if self.desc_type == 'occp':
                return out, torch.cat(desc, 2)
            else:
                return out, c
        else:
            return out


class PatchLocalDecoder(nn.Module):
    ''' Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        #self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None

        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)
    
    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']

        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)

        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class LocalPointDecoder(nn.Module):
    ''' Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val']**2

    def sample_point_feature(self, q, p, fea):
        # q: B x M x 3
        # p: B x N x 3
        # fea: B x N x c_dim
        #p, fea = c
        if self.sample_mode == 'gaussian':
            # distance betweeen each query point to the point cloud
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)**2
            weight = (dist/self.var).exp() # Guassian kernel
        else:
            weight = 1/((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3)+10e-6)

        #weight normalization
        weight = weight/weight.sum(dim=2).unsqueeze(-1)

        c_out = weight @ fea # B x M x c_dim

        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]

        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)

        else:
           if self.c_dim != 0:
                pp, fea = c
                c = self.sample_point_feature(p, pp, fea)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class Recon_Coord(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self,  hidden_size, activation=True): #nfeat, code_nfts, nlevels, num_points,
        super().__init__()
        
        #self.fc_topnet = nn.Linear(hidden_size, code_nfts)
        #self.topnet = TopNet(nfeat, code_nfts, nlevels, num_points)
        #self.decode_net = TopNet(nfeat, code_nfts, nlevels, num_points)
        self.decode_net = Corr_net(hidden_size, activation)

    def forward(self, c):
        # code = self.fc_topnet(c)

        return self.decode_net(c)
       

class TopNet(nn.Module):
    def __init__(self, nfeat, code_nfts, nlevels, num_points, hidden_size):
        super().__init__()
        self.nfeat = nfeat
        self.code_nfts = code_nfts
        self.nin = nfeat + code_nfts
        self.nout = nfeat
        self.tarch = get_arch(nlevels, num_points)
        self.npoints = num_points

        self.fc_topnet = nn.Linear(hidden_size, code_nfts)

        level0 = nn.Sequential(
            mlp(self.code_nfts, [256, 64, self.nfeat * int(self.tarch[0])], bn=False),
            nn.Tanh()
        )
        self.levels = nn.ModuleList([level0])
        nin = self.nin
        for i in range(1, len(self.tarch)):
            if i == len(self.tarch) - 1:
                nout = 3
                bn = False
            else:
                nout = self.nout
                bn = False
                
            level = nn.Sequential(
                self.create_level(i, nin, nout, bn),
                nn.Tanh()
            )
            self.levels.append(level)
            # nin = nout * int(self.tarch[i]) + self.code_nfts
            # print(nin, nout * int(self.tarch[i]))
    
    def create_level(self, level, input_channels, output_channels, bn):
        return mlp_conv(input_channels, [input_channels, int(input_channels / 2),
                                    int(input_channels / 4), int(input_channels / 8),
                                    output_channels * int(self.tarch[level])], bn)
    
    def forward(self, code : torch.Tensor):
        code = self.fc_topnet(code)
        
        nlevels = len(self.tarch)
        level0 = self.levels[0](code).reshape(-1, self.nfeat, int(self.tarch[0]))
        outs = [level0, ]
        for i in range(1, nlevels):
            if i == len(self.tarch) - 1:
                nout = 3
            else:
                nout = self.nout
            inp = outs[-1]
            y = torch.cat([inp, code[:, :, None].expand(-1, -1, inp.shape[2])], 1)
            outs.append(self.levels[i](y).reshape(y.shape[0], nout, -1))
            
        reconstruction = outs[-1].transpose(-1, -2)
        return reconstruction


class Corr_net(nn.Module):
    def __init__(self, hidden_size, activation=True):
        super().__init__()

        if activation:
            self.decoder = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_size+3, out_channels=32, kernel_size=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=8, out_channels=3, kernel_size=1),
                    nn.Tanh(),
                    )
        else:
            self.decoder = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_size+3, out_channels=32, kernel_size=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=16, out_channels=8, kernel_size=1),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=8, out_channels=3, kernel_size=1),
                    )
    

    def forward(self, code : torch.Tensor):
        
        return self.decoder(code.transpose(1, 2)).transpose(2, 1)