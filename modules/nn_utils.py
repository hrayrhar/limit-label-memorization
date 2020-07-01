""" Some tools for building basic NN blocks """
import torch
import numpy as np

from nnlib.nnlib import nn_utils as nnlib_nn_utils
from nnlib.nnlib.nn_utils import infer_shape


def parse_network_from_config(args, input_shape):
    """Parses a neural network from json config."""

    # parse project-specific known networks
    if isinstance(args, dict):
        if args['net'] == 'double-descent-cifar10-resnet18':
            from modules.resnet18_double_descent import make_resnet18k
            net = make_resnet18k(k=args['k'], num_classes=10)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape

    return nnlib_nn_utils.parse_network_from_config(args=args, input_shape=input_shape)


def get_grad_replacement_class(sample=False, standard_dev=None, q_dist='Gaussian'):
    if not sample:
        return GradReplacement

    # otherwise need to define a class that can sample gradients
    class GradReplacementWithSampling(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pred, grad_wrt_logits):
            ctx.save_for_backward(grad_wrt_logits)
            return pred

        @staticmethod
        def backward(ctx, grad_output):
            grad_wrt_logits = ctx.saved_tensors[0]
            if q_dist == 'Gaussian':
                dist = torch.distributions.Normal(loc=grad_wrt_logits, scale=standard_dev)
            elif q_dist == 'Laplace':
                dist = torch.distributions.Laplace(loc=grad_wrt_logits, scale=np.sqrt(1.0/2.0)*standard_dev)
            else:
                raise NotImplementedError()
            return dist.sample(), torch.zeros_like(grad_wrt_logits)

    return GradReplacementWithSampling


class GradReplacement(torch.autograd.Function):
    """ Identity function that gets x and grad_wrt_x and returns x,
    but when returning gradients, returns the given grad_wrt_x instead
    of the correct gradient. This class can be used to replace the true
    gradient with a custom one at any location.
    """
    @staticmethod
    def forward(ctx, pred, grad_wrt_logits):
        ctx.save_for_backward(grad_wrt_logits)
        return pred

    @staticmethod
    def backward(ctx, grad_output):
        grad_wrt_logits = ctx.saved_tensors[0]
        return grad_wrt_logits, torch.zeros_like(grad_wrt_logits)


def get_grad_noise_class(standard_dev=None, q_dist='Gaussian'):
    class GradNoise(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pred):
            return pred

        @staticmethod
        def backward(ctx, grad_output):
            if q_dist == 'Gaussian':
                dist = torch.distributions.Normal(loc=grad_output, scale=standard_dev)
            elif q_dist == 'Laplace':
                dist = torch.distributions.Laplace(loc=grad_output, scale=np.sqrt(1.0/2.0)*standard_dev)
            else:
                raise NotImplementedError()
            return dist.sample()

    return GradNoise
