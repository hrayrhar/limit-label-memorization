from collections import defaultdict
from modules import nn, utils, losses
import numpy as np
import torch
import torch.nn.functional as F


class BaseClassifier(torch.nn.Module):
    """ Abstract class for classifiers.
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__()
        # initialize and use later
        self._accuracy = defaultdict(list)

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
    """ Class standard neural network classifiers.
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

        classifier_loss = F.cross_entropy(input=pred, target=y)

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info


class PretrainedClassifier(BaseClassifier):
    """ Class for classifiers where feature extraction part is pretrained
    in an unsupervised way.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda', **kwargs):
        super(PretrainedClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'class': 'PretrainedClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device

        # initialize the network
        vae = utils.load(self.encoder_path, self.device)
        self.encoder = vae.encoder
        utils.set_requires_grad(self.encoder, False)

        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['label-predictor'],
                                                   input_shape=vae.hidden_shape)
        self.classifier = self.classifier.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        z_params = self.encoder(x)
        z = self.encoder.mean(z_params)  # TODO: maybe sampling is better?

        # compute classifier predictions
        pred = self.classifier(z)

        out = {
            'pred': pred,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        y = labels[0].to(self.device)

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info

# TODO: normalize the last layer parameters?
class PenalizeLastLayerFixedForm(BaseClassifier):
    """ Penalizes the gradients of the last layer and the q network has the correct form.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, weight_decay=0.0,
                 lamb=0.0, **kwargs):
        super(PenalizeLastLayerFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'weight_decay': weight_decay,
            'lamb': lamb,
            'class': 'PenalizeLastLayerFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay
        self.weight_decay = weight_decay
        self.lamb = lamb

        self._current_iteration = defaultdict(lambda: 0)

        # initialize the network
        vae = utils.load(self.encoder_path, self.device)
        self.encoder = vae.encoder
        if self.freeze_encoder:
            utils.set_requires_grad(self.encoder, False)
        else:
            # resent the encoder parameters
            self.encoder.train()
            for layer in self.encoder.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # skip the last linear layer, will add manually
        self.classifier_base, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'][:-1],
                                                        input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)

        self.last_layer = torch.nn.Linear(256, 10, bias=False).to(self.device)

        self.label_predictor, _ = nn.parse_feed_forward(args=self.architecture_args['label-predictor'],
                                                        input_shape=vae.hidden_shape)
        self.label_predictor = self.label_predictor.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        z = self.classifier_base(x)
        pred = self.last_layer(z)

        # predict the gradient wrt to last layer parameters using the correct form
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)  # TODO: maybe sampling is better
        label_pred = self.label_predictor(rx)

        out = {
            'pred': pred,
            'z': z,
            'label_pred': label_pred,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        z = info['z']
        label_pred = info['label_pred']

        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=10).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        info_penalty = self.lamb * torch.sum(z ** 2, dim=1).mean() *\
                       torch.sum((y_one_hot - label_pred) ** 2, dim=1).mean()

        # add weight decay on U, the parameter of self.last_layer
        l2_penalty = 0.0
        for param in self.last_layer.parameters():
            l2_penalty = l2_penalty + torch.sum(param ** 2)
        l2_penalty *= self.weight_decay

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty,
            'l2 penalty': l2_penalty
        }

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PenalizeLastLayerFixedForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                 partition=partition, **kwargs)
        # track some additional statistics
        z = info['z']
        y = batch_labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=10).float()
        label_pred = info['label_pred']

        tensorboard.add_scalar('stats/{}_norm_z'.format(partition),
                               torch.sum(z**2, dim=1).mean(),
                               self._current_iteration[partition])
        tensorboard.add_scalar('stats/{}_norm_label_pred_error'.format(partition),
                               torch.sum((y_one_hot - label_pred) ** 2, dim=1).mean(),
                               self._current_iteration[partition])

        if partition == 'train':
            tensorboard.add_scalar('stats/last_layer_norm',
                                   torch.sum(next(self.last_layer.parameters())**2),
                                   self._current_iteration[partition])

        self._current_iteration[partition] += 1


class PenalizeLastLayerGeneralForm(BaseClassifier):
    """ Penalizes the gradients of the last layer and the q network has the correct form.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, weight_decay=0.0,
                 lamb=0.0, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'weight_decay': weight_decay,
            'lamb': lamb,
            'class': 'PenalizeLastLayerGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay
        self.weight_decay = weight_decay
        self.lamb = lamb

        self._current_iteration = defaultdict(lambda: 0)

        # initialize the network
        vae = utils.load(self.encoder_path, self.device)
        self.encoder = vae.encoder
        if self.freeze_encoder:
            utils.set_requires_grad(self.encoder, False)
        else:
            # resent the encoder parameters
            self.encoder.train()
            for layer in self.encoder.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        # skip the last linear layer, will add manually
        self.classifier_base, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'][:-1],
                                                        input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)

        self.last_layer = torch.nn.Linear(256, 10, bias=False).to(self.device)

        self.grad_predictor = torch.nn.Linear(vae.hidden_shape[-1], 256 * 10).to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        z = self.classifier_base(x)
        pred = self.last_layer(z)

        # predict the gradient wrt to last layer parameters using the general form
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)  # TODO: maybe sampling is better
        grad_pred = self.grad_predictor(rx).reshape((-1, 10, 256))

        out = {
            'pred': pred,
            'z': z,
            'grad_pred': grad_pred,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        z = info['z']
        grad_pred = info['grad_pred']

        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=10).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        grad_actual = y_one_hot.reshape((-1, 10, 1)) * z.reshape((-1, 1, 256))
        info_penalty = self.lamb * ((grad_pred - grad_actual)**2).sum(dim=2).sum(dim=1).mean()

        # add weight decay on U, the parameter of self.last_layer
        l2_penalty = 0.0
        for param in self.last_layer.parameters():
            l2_penalty = l2_penalty + torch.sum(param ** 2)
        l2_penalty *= self.weight_decay

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty,
            'l2 penalty': l2_penalty
        }

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                 partition=partition, **kwargs)
        # track some additional statistics
        z = info['z']

        tensorboard.add_scalar('stats/{}_norm_z'.format(partition),
                               torch.sum(z**2, dim=1).mean(),
                               self._current_iteration[partition])

        if partition == 'train':
            tensorboard.add_scalar('stats/last_layer_norm',
                                   torch.sum(next(self.last_layer.parameters())**2),
                                   self._current_iteration[partition])

        self._current_iteration[partition] += 1


# TODO: add parameter controlling loss weights
# TODO: instead of replacing gradients, maybe we
#       should linearly interpolate between predicted
#       and actual gradients
class RobustClassifier(BaseClassifier):
    """ Class of classifiers where one network predicts the gradients without using
    the labels.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, **kwargs):
        super(RobustClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'class': 'RobustClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay

        # initialize the network
        vae = utils.load(self.encoder_path, self.device)
        self.encoder = vae.encoder
        if self.freeze_encoder:
            utils.set_requires_grad(self.encoder, False)
        else:
            # resent the encoder parameters
            self.encoder.train()
            for layer in self.encoder.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

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
            'grad_wrt_logits': grad_wrt_logits
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        label_pred = info['label_pred']
        grad_wrt_logits = info['grad_wrt_logits']
        y = labels[0].to(self.device)

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # gradient matching loss
        grad_loss = losses.mse(F.one_hot(y, num_classes=10).float(),
                               label_pred)

        batch_losses = {
            'classifier': classifier_loss,
            'grad': grad_loss
        }

        # penalize cases when the predicted gradient is not zero
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_wrt_logits**2, dim=1), dim=0)
            batch_losses['grad_l2'] = grad_l2_loss

        return batch_losses, info
