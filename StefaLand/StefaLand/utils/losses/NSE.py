import torch

class MaskedNSE_hydroDL(torch.nn.Module):
    """
        This file is part of the hydroDL
    """

    def __init__(self, train_y, eps=0.1, device='cpu'):
        super().__init__()
        self.std = np.nanstd(train_y, axis=1)  # train_y: (basin, time, 1)
        self.eps = eps
        self.device = device

    def forward(self, output, target, igrid, batch_first=True):
        if batch_first:
            output = output.transpose(0, 1)  # [seq_len, batch, features]
            target = target.transpose(0, 1)

        nt = target.shape[0]
        stdse = np.tile(self.std[igrid].T, (nt, 1))

        stdbatch = torch.tensor(stdse, requires_grad=False).float().to(self.device)
        p0 = output[:, :, 0]  # dim: Time*Gage
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        stdw = stdbatch[mask]
        sqRes = (p - t) ** 2
        normRes = sqRes / (stdw + self.eps) ** 2
        loss = torch.mean(normRes)

        return loss

class MaskedNSELoss(torch.nn.Module):
    """
    This file is part of the accompanying code to our manuscript:

    Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
    datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
    https://doi.org/10.5194/hess-2020-221, in review, 2020.

    You should have received a copy of the Apache-2.0 license along with the code. If not,
    see <https://opensource.org/licenses/Apache-2.0>
    """

    def __init__(self, eps: float = 0.1):
        super(MaskedNSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, target_stds = None):

        mask = ~torch.isnan(y)
        squared_error = (y_hat[mask] - y[mask])**2

        using_target_stds = target_stds is not None and 0 not in target_stds.size()

        if using_target_stds:
            if target_stds.shape != y_hat.shape:
                target_stds = target_stds.unsqueeze(1).expand_as(y_hat)
            weights = 1 / (target_stds[mask] + self.eps)**2
        else:
            weights = 1.0

        scaled_loss = weights * squared_error

        return torch.mean(scaled_loss)

class NSELoss(torch.nn.Module):
    __name__ = "NSE"

    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):

        if len(target.shape) == 2:
            target = target[:,:,None]

        if len(output.shape) == 2:
            output = output[:,:,None]

        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample + 1
        # minimize the opposite average NSE
        loss = -(losssum / nsample)
        return loss
