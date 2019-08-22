from collections import defaultdict
from modules import nn, utils, losses
import numpy as np
import torch


class BaseClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__()

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy[partition] = []

    def on_iteration_end(self, info, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(info['pred']).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)
        self._accuracy[partition].append((pred == batch_labels).astype(np.float).mean())

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy[partition])
        tensorboard.add_scalar('metrics/{}_accuracy'.format(partition), accuracy, epoch)


class StandardClassifier(BaseClassifier):
    """ Class for creating simple neural networks from architecture configs.
    """
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'device': device,
            'class': 'StandardClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.device = device

        # initialize and use later
        self._accuracy = defaultdict(list)

        # initialize the network
        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)
        pred = self.classifier(x)

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        y = labels[0].to(self.device)

        classifier_loss = torch.nn.functional.cross_entropy(input=pred, target=y)

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info


class RobustClassifier(BaseClassifier):
    """ Class for creating simple neural networks from architecture configs.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda', **kwargs):
        super(RobustClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'class': 'RobustClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device

        # initialize and use later
        self._accuracy = defaultdict(list)

        # initialize the network
        vae = utils.load(self.encoder_path, self.device)
        self.encoder = vae.encoder
        utils.set_requires_grad(self.encoder, False)

        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)

        self.label_predictor, _ = nn.parse_feed_forward(args=self.architecture_args['label-predictor'],
                                                        input_shape=vae.hidden_shape)
        self.label_predictor = self.label_predictor.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred = self.classifier(x)

        # predict the gradient wrt to logits (i.e. the pred variable)
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)  # TODO: maybe sampling is better
        label_pred = self.label_predictor(rx)
        grad_wrt_logits = torch.softmax(pred, dim=1) - label_pred

        # to replace the gradients later we use the following trick
        # this ensures that when we take gradient wrt to pred our
        # predicted grad_wrt_logits will be returned
        pred = nn.GradReplacement.apply(pred, grad_wrt_logits)

        out = {
            'pred': pred,
            'label_pred': label_pred,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        label_pred = info['label_pred']
        y = labels[0].to(self.device)

        # classification loss
        classifier_loss = torch.nn.functional.cross_entropy(input=pred, target=y)

        # gradient matching loss
        grad_loss = losses.mse(torch.nn.functional.one_hot(y, num_classes=10).float(),
                               label_pred)

        batch_losses = {
            'classifier': classifier_loss,
            'grad': grad_loss
        }

        return batch_losses, info
