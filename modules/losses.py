import torch
import torch.nn.functional as F


def binary_cross_entropy(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    ce = F.binary_cross_entropy(pred, target, reduction='none')
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce, dim=0)
    return ce


def mse(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mse = torch.sum((target - pred) ** 2, dim=1)
    mse = torch.mean(mse, dim=0)
    return mse


def mae(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mad = torch.sum(torch.abs(target - pred), dim=1)
    mad = torch.mean(mad, dim=0)
    return mad


def gce(target, pred, q=1.0):
    # generalized cross-entropy loss
    assert q > 1e-6
    pred_y = torch.sum(target * pred).sum(dim=1)
    return torch.mean((1.0 - torch.pow(pred_y, q)) / q, dim=0)


def get_classification_loss(target, pred, loss_function='ce', loss_function_param=None):
    """
    :param target: one-hot encoded vector
    :param pred: predicted logits (i.e. before softmax)
    :param loss_function: 'ce', 'mse', 'mae', 'gce'
    :param loss_function_param: when 'gce' this should specify the q parameter
    """
    if loss_function == 'ce':
        return F.cross_entropy(input=pred, target=target)
    if loss_function == 'mse':
        return mse(target, torch.softmax(pred, dim=1))
    if loss_function == 'mae':
        return mae(target, torch.softmax(pred, dim=1))
    if loss_function == 'gce':
        return gce(target, torch.softmax(pred, dim=1), q=loss_function_param)
    raise NotImplementedError()
