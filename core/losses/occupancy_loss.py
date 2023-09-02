import torch
import torch.nn as nn
import ipdb 

class Occupancy_loss (nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'occup_loss_transp'
        self.weight = weight

    def forward_one(self, occ, occ_labels):
        occ_loss = -1 * (occ_labels * torch.log(occ + 1e-5) + 
                        (1 - occ_labels) * torch.log(1 - occ + 1e-5))

        return occ_loss.mean()

    def forward(self, occ_transp, occup_labels_2, **kw):
        occ_loss_all = self.forward_one(occ_transp, occup_labels_2)

        return occ_loss_all * self.weight


class Occupancy_loss_2 (nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'occup_loss_ori'
        self.weight = weight

    def forward_one(self, occ, occ_labels):
        occ_loss = -1 * (occ_labels * torch.log(occ + 1e-5) + 
                        (1 - occ_labels) * torch.log(1 - occ + 1e-5))

        return occ_loss.mean()

    def forward(self, occ_ori, occup_labels_2, **kw):
        occ_loss_all = self.forward_one(occ_ori, occup_labels_2)

        return occ_loss_all * self.weight



class Recon_loss(nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'recon_loss_transp'
        self.weight = weight

    def forward(self, coords_transp, point_cloud_2, **kw):
        
        dist = torch.cdist(coords_transp, point_cloud_2)
        loss_recon = torch.mean(torch.min(dist, -1)[0] + torch.min(dist, -2)[0])

        return loss_recon * self.weight

class Recon_loss_ori(nn.Module):
    def __init__(self, weight, **kw):
        nn.Module.__init__(self)
        self.name = 'recon_loss_ori'
        self.weight = weight

    def forward(self, coords_ori, point_cloud_1, **kw):
        
        dist = torch.cdist(coords_ori, point_cloud_1)
        loss_recon = torch.mean(torch.min(dist, -1)[0] + torch.min(dist, -2)[0])

        return loss_recon * self.weight