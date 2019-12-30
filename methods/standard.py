from modules import nn_utils, losses, pretrained_models
import torch
import torch.nn.functional as F
from methods import BaseClassifier


class StandardClassifier(BaseClassifier):
    """ Standard classifier trained with cross-entropy loss.
    Has an option to work on pretrained representation of x.
    Optionally, can add noise to the gradient wrt to the output logit.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None,
                 device='cuda', loss_function='ce', add_noise=False, noise_type='Gaussian',
                 noise_std=0.0, **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'loss_function': loss_function,
            'add_noise': add_noise,
            'noise_type': noise_type,
            'noise_std': noise_std,
            'class': 'StandardClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.loss_function = loss_function
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_std = noise_std

        # initialize the network
        self.repr_net = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
        self.repr_shape = self.repr_net.output_shape
        self.classifier, output_shape = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                                    input_shape=self.repr_shape)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(self.device)
        self.grad_noise_class = nn_utils.get_grad_noise_class(standard_dev=noise_std, q_dist=noise_type)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        pred = self.classifier(self.repr_net(x))
        if self.add_noise:
            pred = self.grad_noise_class.apply(pred)

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        y = labels[0].to(self.device)

        # classification loss
        if self.loss_function == 'ce':
            classifier_loss = F.cross_entropy(input=pred, target=y)
        if self.loss_function == 'mse':
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mse(y_one_hot, torch.softmax(pred, dim=1))
        if self.loss_function == 'mae':
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mae(y_one_hot, torch.softmax(pred, dim=1))

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info


class StandardClassifierWithNoise(BaseClassifier):
    """ Standard classifier trained with cross-entropy loss and noisy gradients.
    Has an option to work on pretrained representation of x.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None,
                 device='cuda', loss_function='ce', add_noise=False, noise_type='Gaussian',
                 noise_std=0.0, **kwargs):
        super(StandardClassifierWithNoise, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'loss_function': loss_function,
            'add_noise': add_noise,
            'noise_type': noise_type,
            'noise_std': noise_std,
            'class': 'StandardClassifierWithNoise'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.loss_function = loss_function
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_std = noise_std

        # initialize the network
        self.repr_net = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
        self.repr_shape = self.repr_net.output_shape
        self.classifier, output_shape = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                                    input_shape=self.repr_shape)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        pred = self.classifier(self.repr_net(x))

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        y = labels[0].to(self.device)

        # classification loss
        if self.loss_function == 'ce':
            classifier_loss = F.cross_entropy(input=pred, target=y)
        if self.loss_function == 'mse':
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mse(y_one_hot, torch.softmax(pred, dim=1))
        if self.loss_function == 'mae':
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mae(y_one_hot, torch.softmax(pred, dim=1))

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info

    def before_weight_update(self, **kwargs):
        if not self.add_noise:
            return
        for param in self.parameters():
            if param.requires_grad:
                if self.noise_type == 'Gaussian':
                    param.grad += self.noise_std * torch.randn(size=param.shape, device=self.device)
                else:
                    raise NotImplementedError()
