""" Some tools for building basic NN blocks """
from torch import nn
import torch
import numpy as np

from nnlib.nnlib.nn_utils import infer_shape, add_activation, Flatten, Reshape, Identity
from nnlib.nnlib.networks.conditional_distributions import ConditionalGaussian, ConditionalUniform, \
    ConditionalDiracDelta


def parse_feed_forward(args, input_shape):
    """Parses a sequential feed-forward neural network from json config."""

    # parse known networks
    if isinstance(args, dict):
        if args['net'] == 'resnet-cifar10':
            from nnlib.nnlib.networks.resnet_cifar import resnet34
            net = resnet34(num_classes=10)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape
        if args['net'] == 'resnet-cifar100':
            from nnlib.nnlib.networks.resnet_cifar import resnet34
            net = resnet34(num_classes=100)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape
        if args['net'] == 'resnet-clothing1M':
            from torchvision.models import resnet50
            net = resnet50(num_classes=14)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape
        if args['net'] == 'resnet34-clothing1M':
            from torchvision.models import resnet34
            net = resnet34(num_classes=14)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape
        if args['net'] == 'double-descent-cifar10-resnet18':
            from modules.resnet18_double_descent import make_resnet18k
            net = make_resnet18k(k=args['k'], num_classes=10)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape
        if args['net'] == 'resnet34-imagenet':
            from torchvision.models import resnet34
            net = resnet34(num_classes=1000)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape

    net = []
    for cur_layer in args:
        layer_type = cur_layer['type']
        prev_shape = infer_shape(net, input_shape)
        print(prev_shape)

        if layer_type == 'fc':
            dim = cur_layer['dim']
            assert len(prev_shape) == 2
            net.append(nn.Linear(prev_shape[1], dim))
            if cur_layer.get('batch_norm', False):
                net.append(nn.BatchNorm1d(dim))
            add_activation(net, cur_layer.get('activation', 'linear'))
            if 'dropout' in cur_layer:
                net.append(nn.Dropout(cur_layer['dropout']))

        if layer_type == 'flatten':
            net.append(Flatten())

        if layer_type == 'reshape':
            net.append(Reshape(cur_layer['shape']))

        if layer_type == 'conv':
            assert len(prev_shape) == 4
            net.append(nn.Conv2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'deconv':
            assert len(prev_shape) == 4
            net.append(nn.ConvTranspose2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0),
                output_padding=cur_layer.get('output_padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'identity':
            net.append(Identity())

        if layer_type == 'upsampling':
            net.append(torch.nn.UpsamplingNearest2d(
                scale_factor=cur_layer['scale_factor']
            ))

        if layer_type == 'gaussian':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            mu = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            logvar = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            output_shape = [None, cur_layer['dim']]
            print("output.shape:", output_shape)
            return ConditionalGaussian(net, mu, logvar), output_shape

        if layer_type == 'uniform':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            center = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            radius = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            output_shape = [None, cur_layer['dim']]
            print("output.shape:", output_shape)
            return ConditionalUniform(net, center, radius), output_shape

        if layer_type == 'dirac':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            print("output.shape:", output_shape)
            return ConditionalDiracDelta(net), output_shape

    output_shape = infer_shape(net, input_shape)
    print("output.shape:", output_shape)
    return nn.Sequential(*net), output_shape


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
