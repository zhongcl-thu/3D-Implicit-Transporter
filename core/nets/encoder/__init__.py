from . import (
    pointnet, voxels, pointnetpp
)

# from .. import vnn_occupancy_net_pointnet_dgcnn
from .pointnetpp_attn import *


encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_crop_local_pool': pointnet.PatchLocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'voxel_simple_local': voxels.LocalVoxelEncoder,
    'pointnet_atten': PointNetPlusPlusAttnFusion,
    'pointnet_atten_cross':pointnet.LocalPoolPointnetPPFusion,
}
