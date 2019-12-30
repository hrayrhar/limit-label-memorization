import torch
import torch.nn.functional as F


def mse(x, x_rec):
    x = x.reshape((x.shape[0], -1))
    x_rec = x_rec.reshape((x_rec.shape[0], -1))
    mse = torch.sum((x - x_rec) ** 2, dim=1)
    mse = torch.mean(mse, dim=0)
    return mse


def mae(x, x_rec):
    x = x.reshape((x.shape[0], -1))
    x_rec = x_rec.reshape((x_rec.shape[0], -1))
    mad = torch.sum(torch.abs(x - x_rec), dim=1)
    mad = torch.mean(mad, dim=0)
    return mad


def binary_cross_entropy(x, x_rec):
    x = x.reshape((x.shape[0], -1))
    x_rec = x_rec.reshape((x_rec.shape[0], -1))
    ce = F.binary_cross_entropy(x_rec, x, reduction='none')
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce, dim=0)
    return ce
