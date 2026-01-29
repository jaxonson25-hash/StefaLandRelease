import torch
import torch.nn
from torch import nn

def apply_dropout(m):
    """
    Apply dropout to the model during evaluation or testing phase to enable Monte Carlo Dropout
    model = MonteCarloCNN()
    model.eval()  # Set the whole model to evaluation mode first
    model.apply(apply_dropout)  # Then activate dropout for Monte Carlo simulation
    """
    if isinstance(m, nn.Dropout):
        print("Setting dropout layer to train mode")
        m.train()

def createMask(x, dr, seed):
    mask = x.new().resize_as_(x).bernoulli_(1 - dr).div_(1 - dr).detach_()
    return mask


class DropMask(torch.autograd.function.InplaceFunction):
    @classmethod
    def forward(cls, ctx, input, mask, train=False, inplace=False):
        ctx.train = train
        ctx.inplace = inplace
        ctx.mask = mask

        if not ctx.train:
            return input
        else:
            if ctx.inplace:
                ctx.mark_dirty(input)
                output = input
            else:
                output = input.clone()
            output.mul_(ctx.mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.train:
            return grad_output * ctx.mask, None, None, None
        else:
            return grad_output, None, None, None
