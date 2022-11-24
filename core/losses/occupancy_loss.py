import torch
import torch.nn as nn

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

