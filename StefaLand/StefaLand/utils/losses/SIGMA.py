import torch

"""
code source:
https://zenodo.org/records/4068610
"""


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior_distribution='Gauss'):
        super().__init__()
        self.prior_distribution = prior_distribution.split('+')

    # def forward(self, input, target):
    #     pred = input[:, :, 0]
    #     noise = input[:, :, 1]
    #
    #     obs = target[:, :, 0]
    #     loc0 = obs == pred
    #     noise[loc0] = 1

    def forward(self, pred, noise, obs):

        if self.prior_distribution[0] == 'Gauss':
            loss = torch.exp(-noise).mul(torch.mul(pred - obs, pred - obs)) / 2 + noise / 2
            lossMeanT = torch.mean(loss, dim=0)

        # elif self.prior_distribution[0] == 'invGamma':
        #     c1 = float(self.prior_distribution[1])
        #     c2 = float(self.prior_distribution[2])
        #     nt = pred.shape[0]
        #     loss = torch.exp(-noise).mul(torch.mul(pred - obs, pred - obs) + c2 / nt) / 2 + (1 / 2 + c1 / nt) * noise
        #     loss[loc0] = 0
        #     lossMeanT = torch.mean(loss, dim=0)

        lossMeanB = torch.mean(lossMeanT)
        return lossMeanB
