""" Some tools for building basic NN blocks """
import numpy as np
from torch import nn
import torch


def infer_shape(layers, input_shape):
    """Given a list of layers representing a sequential model and its input_shape, infers the output shape."""
    input_shape = [x for x in input_shape]
    if input_shape[0] is None:
        input_shape[0] = 4  # should be more than 1, otherwise batch norm will not work
    x = torch.tensor(np.random.normal(size=input_shape), dtype=torch.float, device='cpu')
    for layer in layers:
        x = layer(x)
    output_shape = list(x.shape)
    output_shape[0] = None
    return output_shape


def add_activation(layers, activation):
    """Adds an activation function into a list of layers."""
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    if activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    if activation == 'tanh':
        layers.append(nn.Tanh())
    if activation == 'softplus':
        layers.append(nn.Softplus())
    if activation == 'linear':
        pass
    return layers


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self._shape = tuple([-1, ] + list(shape))

    def forward(self, x):
        return x.view(self._shape)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def parse_feed_forward(args, input_shape):
    """Parses a sequential feed-forward neural network from json config."""
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


# Conditional distributions should return list of parameters.
# They should have these functions defined:
# - sample(params)
# - log_probs(params, z)
# - mean(params)
# - log_marginal_probs(params, z)
# - kl_divergence

class ConditionalGaussian(nn.Module):
    """Conditional Gaussian distribution, where the mean and variance are
    parametrized with neural networks."""
    def __init__(self, net, mu, logvar):
        super(ConditionalGaussian, self).__init__()
        self.net = net
        self.mu = mu
        self.logvar = logvar

    def forward(self, x):
        h = self.net(x)
        return {
            'param:mu': self.mu(h),
            'param:logvar': self.logvar(h)
        }

    @staticmethod
    def mean(params):
        return params['param:mu']

    @staticmethod
    def sample(params):
        mu = params['param:mu']
        logvar = params['param:logvar']
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def log_prob(params, z):
        return ConditionalGaussian.log_marginal_probs(params, z).sum(dim=-1)

    @staticmethod
    def log_marginal_probs(params, z):
        mu = params['param:mu']
        logvar = params['param:logvar']
        return -0.5 * (np.log(2 * np.pi) + logvar + ((z - mu) ** 2) / torch.exp(logvar))

    @staticmethod
    def kl_divergence(params):
        """ Computes KL(q(z|x) || p(z)) assuming p(z) is N(0, I). """
        mu = params['param:mu']
        logvar = params['param:logvar']
        kl = -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar), dim=1)
        return torch.mean(kl, dim=0)


class ConditionalUniform(nn.Module):
    """Conditional uniform distribution parametrized with a NN. """
    def __init__(self, net, center, radius):
        super(ConditionalUniform, self).__init__()
        self.net = net
        self.center = center
        self.radius = radius

    def forward(self, x):
        h = self.net(x)
        c = torch.tanh(self.center(h))
        r = torch.sigmoid(self.radius(h))
        return {
            'param:left': torch.clamp(c - r, -1, +1),
            'param:right': torch.clamp(c + r, -1, +1)
        }

    @staticmethod
    def mean(params):
        return 0.5 * (params['param:left'] + params['param:right'])

    @staticmethod
    def sample(params):
        left = params['param:left']
        right = params['param:right']
        eps = torch.rand_like(left)
        return left + (right - left) * eps

    @staticmethod
    def log_prob(params, z):
        return ConditionalUniform.log_marginal_probs(params, z).sum(dim=-1)

    @staticmethod
    def log_marginal_probs(params, z):
        return -torch.log(params['param:right'] - params['param:left'] + 1e-6)

    @staticmethod
    def kl_divergence(params):
        """ Computes KL(q(z|x) || p(z)) assuming p(z) is U(-1, +1). """
        left = params['param:left']
        right = params['param:right']
        kl = torch.sum(torch.log(2.0 / (right - left + 1e-6)), dim=1)
        return torch.mean(kl, dim=0)


class ConditionalDiracDelta(nn.Module):
    """Conditional Dirac delta distribution parametrized with a NN.
    This can be used to make the VAE class act like a regular AE. """
    def __init__(self, net):
        super(ConditionalDiracDelta, self).__init__()
        self.net = net

    def forward(self, x):
        return {
            'param:mu': self.net(x)
        }

    @staticmethod
    def mean(params):
        return params['param:mu']

    @staticmethod
    def sample(params):
        return params['param:mu']

    @staticmethod
    def log_prob(params, z):
        return ValueError('You cannot call log_prob for Dirac delta.')

    @staticmethod
    def log_marginal_probs(params, z):
        return ValueError('You cannot call log_marginal_probs for Dirac delta.')

    @staticmethod
    def kl_divergence(params):
        return 0
