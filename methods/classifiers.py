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
        self._current_iteration = defaultdict(lambda: 0)

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

        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['2layer-classifier'],
                                                   input_shape=vae.hidden_shape)
        self.classifier = self.classifier.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        z_params = self.encoder(x)
        z = self.encoder.mean(z_params)

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
                 freeze_encoder=True, weight_decay=0.0, lamb=0.0, **kwargs):
        super(PenalizeLastLayerFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
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
        self.weight_decay = weight_decay
        self.lamb = lamb

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

        self.label_predictor, _ = nn.parse_feed_forward(args=self.architecture_args['2layer-classifier'],
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
        rx = self.encoder.mean(rx_params)
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
                       losses.mse(y_one_hot, label_pred)

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
                               losses.mse(y_one_hot, label_pred),
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
                 freeze_encoder=True, weight_decay=0.0, lamb=0.0, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
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
        self.weight_decay = weight_decay
        self.lamb = lamb

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
        rx = self.encoder.mean(rx_params)
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
        info_penalty = self.lamb * losses.mse(grad_pred, grad_actual)

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


# TODO: instead of replacing gradients, maybe we
#       should linearly interpolate between predicted
#       and actual gradients
class PredictGradOutputFixedForm(BaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network uses the form of output gradients.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, lamb=0.0, **kwargs):
        super(PredictGradOutputFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'lamb': 'lamb',
            'class': 'PredictGradOutputFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

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

        self.label_predictor, _ = nn.parse_feed_forward(args=self.architecture_args['2layer-classifier'],
                                                        input_shape=vae.hidden_shape)
        self.label_predictor = self.label_predictor.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred = self.classifier(x)

        # predict the gradient wrt to logits
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)
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
        y_one_hot = F.one_hot(y, num_classes=10).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # gradient matching loss
        grad_loss = self.lamb * losses.mse(y_one_hot, label_pred)

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


class PredictGradOutputGeneralForm(BaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, lamb=0.0, **kwargs):
        super(PredictGradOutputGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'lamb': 'lamb',
            'class': 'PredictGradOutputGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

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

        self.grad_predictor = torch.nn.Linear(vae.hidden_shape[-1], 10).to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred_before = self.classifier(x)

        # predict the gradient wrt to logits
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)
        grad_pred = self.grad_predictor(rx)

        # to replace the gradients later we use the following trick
        # this ensures that when we take gradient wrt to pred our
        # predicted grad_pred will be returned
        pred = nn.GradReplacement.apply(pred_before, grad_pred)

        out = {
            'pred': pred,
            'pred_before': pred_before,
            'grad_pred': grad_pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        pred_before = info['pred_before']
        grad_pred = info['grad_pred']
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=10).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # gradient matching loss
        grad_actual = torch.softmax(pred_before, dim=1) - y_one_hot
        grad_loss = self.lamb * losses.mse(grad_actual, grad_pred)

        batch_losses = {
            'classifier': classifier_loss,
            'grad': grad_loss
        }

        # penalize cases when the predicted gradient is not zero
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PredictGradOutputGeneralForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                   partition=partition, **kwargs)
        # track some additional statistics
        grad_pred = info['grad_pred']

        tensorboard.add_scalar('stats/{}_pred_grad_norm'.format(partition),
                               torch.sum(grad_pred**2, dim=1).mean(),
                               self._current_iteration[partition])

        self._current_iteration[partition] += 1


class PredictGradOutputMetaLearning(BaseClassifier):
    """ Trains the classifier using predicted gradients. The gradient predictor has general form and
    we train it using meta learning type objective.
    """
    def __init__(self, input_shape, architecture_args, encoder_path, device='cuda',
                 freeze_encoder=True, grad_weight_decay=0.0, nsteps=1, **kwargs):
        super(PredictGradOutputMetaLearning, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'encoder_path': encoder_path,
            'device': device,
            'freeze_encoder': freeze_encoder,
            'grad_weight_decay': grad_weight_decay,
            'nsteps': nsteps,
            'class': 'PredictGradOutputMetaLearning'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.encoder_path = encoder_path
        self.device = device
        self.freeze_encoder = freeze_encoder
        self.grad_weight_decay = grad_weight_decay
        self.nsteps = nsteps
        self.grad_diff_coef = 100.0

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

        self.grad_predictor = torch.nn.Linear(vae.hidden_shape[-1], 10).to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)

        # predict the gradient wrt to logits
        rx_params = self.encoder(x)
        rx = self.encoder.mean(rx_params)
        grad_pred = self.grad_predictor(rx)

        out = {
            'grad_pred': grad_pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        grad_pred = info['grad_pred']

        x = inputs[0].to(self.device)
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=10).float()

        # ====================== meta-training ======================
        # detach classifier params and initialize again
        # TODO: learn initial weights?
        self.classifier.zero_grad()
        for param in self.classifier.parameters():
            param.detach_()
            param.requires_grad_()
        # TODO: maybe always initialize to the same value?
        # TODO: reset?
        # for layer in self.classifier.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()

        torch.set_grad_enabled(True)  # enable tracking if it is disabled
        lr = 0.0001  # TODO: tune or learn?
        chunk_size = x.shape[0] // (self.nsteps + 1)

        for idx in range(self.nsteps):
            start = chunk_size * idx
            end = start + chunk_size

            # compute classifier predictions
            pred_before = self.classifier(x[start:end])

            # to replace the gradients later we use the following trick
            # this ensures that when we take gradient wrt to pred our
            # predicted grad_pred will be returned
            pred = nn.GradReplacement.apply(pred_before, grad_pred[start:end])

            # compute all gradients with respect to classifier parameters
            # and do a single stochastic gradient descent update
            classifier_loss = F.cross_entropy(input=pred, target=y[start:end])
            classifier_grads = torch.autograd.grad(classifier_loss, self.classifier.parameters(),
                                                   create_graph=True)

            updated_state_dict = dict()
            for (param_name, param), grad in zip(list(self.classifier.named_parameters()),
                                                 classifier_grads):
                updated_state_dict[param_name] = param - lr * grad

            for param_full_name, param in updated_state_dict.items():
                module_id = param_full_name[:param_full_name.index('.')]
                param_name = param_full_name[param_full_name.find('.') + 1:]
                self.classifier._modules[module_id]._parameters.pop(param_name)
                self.classifier._modules[module_id]._parameters[param_name] = param

        # ====================== meta-testing ======================
        torch.set_grad_enabled(grad_enabled)
        start = self.nsteps * chunk_size
        end = x.shape[0]

        # compute final classifier loss
        pred_before = self.classifier(x)
        pred = nn.GradReplacement.apply(pred_before, grad_pred)
        info['pred'] = pred_before  # for computing accuracy-like statistics

        # classifier loss on the last chunk
        classifier_loss = F.cross_entropy(input=pred[start:end], target=y[start:end])

        # help the method by saying that it should output cross-entropy like gradients
        grad_diff_loss = self.grad_diff_coef * losses.mse(grad_pred,
                                                          torch.softmax(pred_before.detach(), dim=1) - y_one_hot)

        batch_losses = {
            'classifier': classifier_loss,
            'grad_diff': grad_diff_loss
        }

        # penalize cases when the predicted gradient is not zero
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1))
            batch_losses['grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PredictGradOutputMetaLearning, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                    partition=partition, **kwargs)
        # track some additional statistics
        grad_pred = info['grad_pred']

        tensorboard.add_scalar('stats/{}_pred_grad_norm'.format(partition),
                               torch.sum(grad_pred**2, dim=1).mean(),
                               self._current_iteration[partition])

        self._current_iteration[partition] += 1

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        super(PredictGradOutputMetaLearning, self).on_epoch_end(partition=partition,
                                                                epoch=epoch,
                                                                tensorboard=tensorboard)
        self.grad_diff_coef *= 0.9
        if partition == 'train':
            tensorboard.add_scalar('hyperparameters/grad_diff_coef',
                                   self.grad_diff_coef,
                                   epoch)
