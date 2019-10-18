from collections import defaultdict
from modules import nn, utils, losses
from modules import visualization as vis
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
        self._current_iteration[partition] += 1
        pred = utils.to_numpy(info['pred']).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)
        self._accuracy[partition].append((pred == batch_labels).astype(np.float).mean())

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy[partition])
        tensorboard.add_scalar('metrics/{}_accuracy'.format(partition), accuracy, epoch)

    def forward(self, *input, **kwargs):
        raise NotImplementedError()

    def compute_loss(self, *input, **kwargs):
        raise NotImplementedError()

    def visualize(self, train_loader, val_loader, tensorboard, epoch, **kwargs):
        visualizations = {}

        # add gradient norm histograms
        vis.ce_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-ce-grad')
        if val_loader is not None:
            vis.ce_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-ce-grad')

        # add gradient pair plots
        for p in [(0, 1), (4, 9)]:
            fig, _ = vis.ce_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
            visualizations['gradients/train-ce-scatter-{}-{}'.format(p[0], p[1])] = fig
            if val_loader is not None:
                fig, _ = vis.ce_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                visualizations['gradients/val-ce-scatter-{}-{}'.format(p[0], p[1])] = fig

        return visualizations


class StandardClassifier(BaseClassifier):
    """ Standard classifier trained with cross-entropy loss.
    Has an option to use a pretrained VAE as feature extractor.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path=None,
                 freeze_pretrained_parts=False, device='cuda', loss_function='ce',
                 **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'device': device,
            'loss_function': loss_function,
            'class': 'StandardClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.loss_function = loss_function

        # initialize the network
        self.num_classes = self.architecture_args['classifier'][-1]['dim']
        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding layers of the classifier
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.classifier.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            classifier_params = dict(self.classifier.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                classifier_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        out = {
            'pred': self.classifier(x),
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
            y_one_hot = y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mse(y_one_hot, torch.softmax(pred, dim=1))
        if self.loss_function == 'mad':
            y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
            classifier_loss = losses.mad(y_one_hot, torch.softmax(pred, dim=1))

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, info


class PenalizeLastLayerFixedForm(BaseClassifier):
    """ Penalizes the gradients of the last layer weights. The q network has the correct form:
    (s(a) - y) z^T. Therefore, with q predicts y.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path=None, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, lamb=1.0, **kwargs):
        super(PenalizeLastLayerFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'lamb': lamb,
            'class': 'PenalizeLastLayerFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

        # initialize the network
        self.num_classes = self.architecture_args['classifier'][-1]['dim']
        self.last_layer_dim = self.architecture_args['classifier'][-2]['dim']

        self.classifier_base, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'][:-1],
                                                        input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)
        self.classifier_last_layer = torch.nn.Linear(self.last_layer_dim,
                                                     self.num_classes,
                                                     bias=False).to(self.device)

        self.q_base, q_base_shape = nn.parse_feed_forward(args=self.architecture_args['q-base'],
                                                          input_shape=self.input_shape)
        self.q_base = self.q_base.to(self.device)

        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1], 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding q_base
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.q_base.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            q_base_params = dict(self.q_base.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                q_base_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        z = self.classifier_base(x)
        pred = self.classifier_last_layer(z)

        # predict labels from x
        q_label_pred = self.q_top(self.q_base(x))

        out = {
            'pred': pred,
            'z': z,
            'q_label_pred': q_label_pred,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        z = info['z']
        q_label_pred = info['q_label_pred']

        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        info_penalty = self.lamb * torch.mean(torch.sum(z ** 2, dim=1) *
                                              torch.sum((y_one_hot - torch.softmax(q_label_pred, dim=1))**2, dim=1),
                                              dim=0)

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            # predicted gradient is [s(a) - s(q_label_pred)] z^T
            diff = torch.softmax(pred, dim=1) - torch.softmax(q_label_pred, dim=1)
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(z ** 2, dim=1) * torch.sum(diff ** 2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PenalizeLastLayerFixedForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                 partition=partition, **kwargs)
        # track some additional statistics
        tensorboard.add_scalar('stats/{}_norm_z'.format(partition),
                               torch.sum(info['z']**2, dim=1).mean(),
                               self._current_iteration[partition])


class PenalizeLastLayerGeneralForm(BaseClassifier):
    """ Penalizes the gradients of the last layer weights. The q network has general form.
    q(g | x, W) = Net() where g = dL/dU with U being the parameters of the last layer.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, lamb=1.0, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'lamb': lamb,
            'class': 'PenalizeLastLayerGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

        # initialize the network
        self.num_classes = self.architecture_args['classifier'][-1]['dim']
        self.last_layer_dim = self.architecture_args['classifier'][-2]['dim']

        self.classifier_base, _ = nn.parse_feed_forward(
            args=self.architecture_args['classifier'][:-1],
            input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)
        self.classifier_last_layer = torch.nn.Linear(self.last_layer_dim,
                                                     self.num_classes,
                                                     bias=False).to(self.device)

        self.q_base, q_base_shape = nn.parse_feed_forward(args=self.architecture_args['q-base'],
                                                          input_shape=self.input_shape)
        self.q_base = self.q_base.to(self.device)

        # TODO: parametrize better. Takes (r(x), z), outputs matrix.
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + self.last_layer_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes * self.last_layer_dim)).to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding q_base
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.q_base.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            q_base_params = dict(self.q_base.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                q_base_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        z = self.classifier_base(x)
        pred = self.classifier_last_layer(z)

        # predict the gradient wrt to last layer parameters using the general form
        rx = self.q_base(x)
        grad_pred = self.q_top(torch.cat([rx, z], dim=1))
        grad_pred = grad_pred.reshape((-1, self.num_classes, self.last_layer_dim))

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
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        error = (torch.softmax(pred, dim=1) - y_one_hot).reshape(-1, self.num_classes, 1)
        grad_actual = error * z.reshape((-1, 1, self.last_layer_dim))
        info_penalty = self.lamb * losses.mse(grad_pred, grad_actual)

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty,
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            # predicted gradient is grad_pred
            grad_l2_loss = self.grad_weight_decay *\
                           torch.sum((grad_pred - grad_actual)**2, dim=1).sum(dim=1).mean(dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                   partition=partition, **kwargs)
        # track some additional statistics
        tensorboard.add_scalar('stats/{}_norm_z'.format(partition),
                               torch.sum(info['z']**2, dim=1).mean(),
                               self._current_iteration[partition])


class PredictGradOutputFixedForm(BaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network uses the form of output gradients.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, lamb=1.0, **kwargs):
        super(PredictGradOutputFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'lamb': 'lamb',
            'class': 'PredictGradOutputFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

        # initialize the network
        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        self.q_base, q_base_shape = nn.parse_feed_forward(args=self.architecture_args['q-base'],
                                                          input_shape=self.input_shape)
        self.q_base = self.q_base.to(self.device)

        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1], 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding q_base
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.q_base.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            q_base_params = dict(self.q_base.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                q_base_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred = self.classifier(x)

        # predict the gradient wrt to logits
        q_label_pred = self.q_top(self.q_base(x))
        grad_pred = torch.softmax(pred, dim=1) - torch.softmax(q_label_pred, dim=1)

        # change the gradients
        pred = nn.GradReplacement.apply(pred, grad_pred)

        out = {
            'pred': pred,
            'q_label_pred': q_label_pred,
            'grad_pred': grad_pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        q_label_pred = info['q_label_pred']
        grad_pred = info['grad_pred']
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        info_penalty = self.lamb * losses.mse(y_one_hot, torch.softmax(q_label_pred, dim=1))

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, info

    def visualize(self, train_loader, val_loader, tensorboard, epoch, **kwargs):
        visualizations = super(PredictGradOutputFixedForm, self).visualize(train_loader, val_loader, tensorboard, epoch)

        # add gradient norm histograms
        vis.pred_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-pred-grad')
        if val_loader is not None:
            vis.pred_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-pred-grad')

        # add gradient pair plots
        for p in [(0, 1), (4, 9)]:
            fig, _ = vis.pred_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
            visualizations['gradients/train-pred-scatter-{}-{}'.format(p[0], p[1])] = fig
            if val_loader is not None:
                fig, _ = vis.pred_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                visualizations['gradients/val-pred-scatter-{}-{}'.format(p[0], p[1])] = fig

        return visualizations


class PredictGradOutputGeneralForm(BaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, lamb=1.0, **kwargs):
        super(PredictGradOutputGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'lamb': 'lamb',
            'class': 'PredictGradOutputGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

        # initialize the network
        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        self.q_base, q_base_shape = nn.parse_feed_forward(args=self.architecture_args['q-base'],
                                                          input_shape=self.input_shape)
        self.q_base = self.q_base.to(self.device)

        # NOTE: we want to use classifier parameters too
        # TODO: find a good parametrization
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + self.num_classes, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding q_base
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.q_base.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            q_base_params = dict(self.q_base.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                q_base_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred_before = self.classifier(x)

        # predict the gradient wrt to logits
        rx = self.q_base(x)
        grad_pred = self.q_top(torch.cat([rx, pred_before], dim=1))

        # change the gradients
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
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        grad_actual = torch.softmax(pred_before, dim=1) - y_one_hot
        info_penalty = self.lamb * losses.mse(grad_actual, grad_pred)

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PredictGradOutputGeneralForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                   partition=partition, **kwargs)
        # track some additional statistics
        grad_pred = info['grad_pred']

        tensorboard.add_scalar('stats/{}_pred_grad_norm'.format(partition),
                               torch.sum(grad_pred**2, dim=1).mean(),
                               self._current_iteration[partition])

    def visualize(self, train_loader, val_loader, tensorboard, epoch, **kwargs):
        visualizations = super(PredictGradOutputFixedForm, self).visualize(train_loader, val_loader, tensorboard, epoch)

        # add gradient norm histograms
        vis.pred_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-pred-grad')
        if val_loader is not None:
            vis.pred_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-pred-grad')

        # add gradient pair plots
        for p in [(0, 1), (4, 9)]:
            fig, _ = vis.pred_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
            visualizations['gradients/train-pred-scatter-{}-{}'.format(p[0], p[1])] = fig
            if val_loader is not None:
                fig, _ = vis.pred_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                visualizations['gradients/val-pred-scatter-{}-{}'.format(p[0], p[1])] = fig

        return visualizations


class PredictGradOutputGeneralFormUseLabel(BaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form and uses label information.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, lamb=1.0, **kwargs):
        super(PredictGradOutputGeneralFormUseLabel, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'lamb': 'lamb',
            'class': 'PredictGradOutputGeneralFormUseLabel'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb

        # initialize the network
        self.classifier, _ = nn.parse_feed_forward(args=self.architecture_args['classifier'],
                                                   input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        self.q_base, q_base_shape = nn.parse_feed_forward(args=self.architecture_args['q-base'],
                                                          input_shape=self.input_shape)
        self.q_base = self.q_base.to(self.device)

        # NOTE: we want to use classifier parameters too
        # TODO: find a good parametrization
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + 2 * self.num_classes, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

        if pretrained_vae_path is not None:
            # use the vae encoder weights to initialize corresponding q_base
            vae = utils.load(self.pretrained_vae_path, self.device)
            self.q_base.load_state_dict(vae.encoder.net.state_dict(), strict=False)
            q_base_params = dict(self.q_base.named_parameters())
            for name, param in vae.encoder.net.named_parameters():
                q_base_params[name].requires_grad = not freeze_pretrained_parts

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred_before = self.classifier(x)

        out = {
            'pred_before': pred_before,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred_before = info['pred_before']

        x = inputs[0].to(self.device)
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # predict the gradient wrt to logits
        rx = self.q_base(x)
        grad_pred = self.q_top(torch.cat([rx, pred_before, y_one_hot], dim=1))
        info['grad_pred'] = grad_pred

        # change the gradients
        pred = nn.GradReplacement.apply(pred_before, grad_pred)
        info['pred'] = pred

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # gradient matching loss
        grad_actual = torch.softmax(pred_before, dim=1) - y_one_hot
        grad_diff = self.lamb * losses.mse(grad_actual, grad_pred)

        batch_losses = {
            'classifier': classifier_loss,
            'grad_diff': grad_diff
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PredictGradOutputGeneralFormUseLabel, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                   partition=partition, **kwargs)
        # track some additional statistics
        grad_pred = info['grad_pred']

        tensorboard.add_scalar('stats/{}_pred_grad_norm'.format(partition),
                               torch.sum(grad_pred**2, dim=1).mean(),
                               self._current_iteration[partition])

    def visualize(self, train_loader, val_loader, tensorboard, epoch, **kwargs):
        visualizations = super(PredictGradOutputFixedForm, self).visualize(train_loader, val_loader, tensorboard, epoch)

        # add gradient norm histograms
        vis.pred_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-pred-grad')
        if val_loader is not None:
            vis.pred_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-pred-grad')

        # add gradient pair plots
        for p in [(0, 1), (4, 9)]:
            fig, _ = vis.pred_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
            visualizations['gradients/train-pred-scatter-{}-{}'.format(p[0], p[1])] = fig
            if val_loader is not None:
                fig, _ = vis.pred_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                visualizations['gradients/val-pred-scatter-{}-{}'.format(p[0], p[1])] = fig

        return visualizations


# TODO: use weights when predicting gradients
# TODO: use deep supervision
class PredictGradOutputMetaLearning(BaseClassifier):
    """ Trains the classifier using predicted gradients. The gradient predictor has general form and
    we train it using meta learning type objective.
    """
    def __init__(self, input_shape, architecture_args, pretrained_vae_path, device='cuda',
                 freeze_pretrained_parts=True, grad_weight_decay=0.0, nsteps=1, **kwargs):
        super(PredictGradOutputMetaLearning, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_vae_path': pretrained_vae_path,
            'device': device,
            'freeze_pretrained_parts': freeze_pretrained_parts,
            'grad_weight_decay': grad_weight_decay,
            'nsteps': nsteps,
            'class': 'PredictGradOutputMetaLearning'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_vae_path = pretrained_vae_path
        self.device = device
        self.freeze_pretrained_parts = freeze_pretrained_parts
        self.grad_weight_decay = grad_weight_decay
        self.nsteps = nsteps
        self.grad_diff_coef = 100.0

        # initialize the network
        vae = utils.load(self.pretrained_vae_path, self.device)
        self.encoder = vae.encoder
        if self.freeze_pretrained_parts:
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
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        self.grad_predictor = torch.nn.Linear(vae.hidden_shape[-1], self.num_classes).to(self.device)

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
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

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

        # add predicted gradient norm penalty
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

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        super(PredictGradOutputMetaLearning, self).on_epoch_end(partition=partition,
                                                                epoch=epoch,
                                                                tensorboard=tensorboard)
        self.grad_diff_coef *= 0.9
        if partition == 'train':
            tensorboard.add_scalar('hyperparameters/grad_diff_coef',
                                   self.grad_diff_coef,
                                   epoch)
