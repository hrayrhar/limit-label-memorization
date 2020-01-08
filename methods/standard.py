from modules import nn_utils, losses, pretrained_models, utils, baseline_utils
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
                 noise_std=0.0, loss_function_param=None, load_from=None, **kwargs):
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
            'loss_function_param': loss_function_param,
            'load_from': load_from,
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
        self.loss_function_param = loss_function_param
        self.load_from = load_from

        # initialize the network
        self.repr_net = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
        self.repr_shape = self.repr_net.output_shape
        self.classifier, output_shape = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                                    input_shape=self.repr_shape)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(self.device)
        self.grad_noise_class = nn_utils.get_grad_noise_class(standard_dev=noise_std, q_dist=noise_type)

        if self.load_from is not None:
            print("Loading the classifier model from {}".format(load_from))
            stored_net = utils.load(load_from, device='cpu')
            stored_net_params = dict(stored_net.classifier.named_parameters())
            for key, param in self.classifier.named_parameters():
                param.data = stored_net_params[key].data.to(self.device)

        print(self)

    def on_epoch_start(self, partition, epoch, loader, **kwargs):
        super(StandardClassifier, self).on_epoch_start(partition=partition, epoch=epoch,
                                                       loader=loader, **kwargs)

        # In case of FW model, estimate the transition matrix and pass it to the model
        if partition == 'train' and epoch == 0 and self.loss_function == 'fw':
            T_est = baseline_utils.estimate_transition(load_from=self.load_from, data_loader=loader,
                                                       device=self.device)
            self.loss_function_param = T_est

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
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(target=y_one_hot, pred=pred,
                                                         loss_function=self.loss_function,
                                                         loss_function_param=self.loss_function_param)

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
                 noise_std=0.0, loss_function_param=None, **kwargs):
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
            'loss_function_param': loss_function_param,
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
        self.loss_function_param = loss_function_param

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
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(target=y_one_hot, pred=pred,
                                                         loss_function=self.loss_function,
                                                         loss_function_param=self.loss_function_param)

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
