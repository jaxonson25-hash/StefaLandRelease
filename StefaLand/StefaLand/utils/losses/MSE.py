"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Hochreiter, S., and Nearing, G. S.: A note on leveraging synergy in multiple meteorological
datasets with deep learning for rainfall-runoff modeling, Hydrol. Earth Syst. Sci. Discuss.,
https://doi.org/10.5194/hess-2020-221, in review, 2020.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import numpy as np
import torch

ONE_OVER_2PI_SQUARED = 1.0 / np.sqrt(2.0 * np.pi)


class MaskedMSELoss(torch.nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, **kwargs):
        mask = ~torch.isnan(y)
        loss = 0.5 * torch.mean((y_hat[mask] - y[mask])**2)
        return loss