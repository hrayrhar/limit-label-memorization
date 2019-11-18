from collections import defaultdict
from modules import nn_utils, utils, losses, pretrained_models
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

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = {}

        # gradient norm tensorboard histograms
        if tensorboard is not None:
            vis.ce_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-ce-grad')
            if val_loader is not None:
                vis.ce_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-ce-grad')

        # add gradient pair plots
        if train_loader.dataset.dataset_name == 'mnist':
            for p in [(0, 1), (4, 9)]:
                fig, _ = vis.ce_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
                visualizations['gradients/train-ce-scatter-{}-{}'.format(p[0], p[1])] = fig
                if val_loader is not None:
                    fig, _ = vis.ce_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                    visualizations['gradients/val-ce-scatter-{}-{}'.format(p[0], p[1])] = fig

        # visualize pred
        fig, _ = vis.plot_predictions(self, train_loader, key='pred')
        visualizations['predictions/pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='pred')
            visualizations['predictions/pred-val'] = fig

        return visualizations


class StandardClassifier(BaseClassifier):
    """ Standard classifier trained with cross-entropy loss.
    Has an option to work on pretrained representation of x.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None,
                 device='cuda', loss_function='ce',
                 **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'loss_function': loss_function,
            'class': 'StandardClassifier'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.loss_function = loss_function

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

        out = {
            'pred': self.classifier(self.repr_net(x)),
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
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, **kwargs):
        super(PenalizeLastLayerFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': lamb,
            'class': 'PenalizeLastLayerFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb

        # initialize the network
        self.num_classes = self.architecture_args['classifier'][-1]['dim']
        self.last_layer_dim = self.architecture_args['classifier'][-2]['dim']

        self.classifier_base, _ = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'][:-1],
                                                              input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)
        self.classifier_last_layer = torch.nn.Linear(self.last_layer_dim,
                                                     self.num_classes,
                                                     bias=False).to(self.device)

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1], 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

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

        if self.grad_l1_penalty > 0:
            raise NotImplementedError("Implement L1 penalty in this case.")

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
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': lamb,
            'class': 'PenalizeLastLayerGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb

        # initialize the network
        self.num_classes = self.architecture_args['classifier'][-1]['dim']
        self.last_layer_dim = self.architecture_args['classifier'][-2]['dim']

        self.classifier_base, _ = nn_utils.parse_feed_forward(
            args=self.architecture_args['classifier'][:-1],
            input_shape=self.input_shape)
        self.classifier_base = self.classifier_base.to(self.device)
        self.classifier_last_layer = torch.nn.Linear(self.last_layer_dim,
                                                     self.num_classes,
                                                     bias=False).to(self.device)

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        # TODO: parametrize better. Takes (r(x), z), outputs matrix.
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + self.last_layer_dim, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes * self.last_layer_dim)).to(self.device)

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

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty *\
                           torch.sum(torch.abs(grad_pred - grad_actual), dim=1).sum(dim=1).mean(dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, info

    def on_iteration_end(self, info, batch_labels, partition, tensorboard, **kwargs):
        super(PenalizeLastLayerGeneralForm, self).on_iteration_end(info=info, batch_labels=batch_labels,
                                                                   partition=partition, **kwargs)
        # track some additional statistics
        tensorboard.add_scalar('stats/{}_norm_z'.format(partition),
                               torch.sum(info['z']**2, dim=1).mean(),
                               self._current_iteration[partition])


class PredictGradBaseClassifier(BaseClassifier):
    """ Abstract class for gradient prediction approaches.
    """
    def __init__(self, **kwargs):
        super(PredictGradBaseClassifier, self).__init__(**kwargs)

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = super(PredictGradBaseClassifier, self).visualize(
            train_loader, val_loader, tensorboard, epoch)

        # gradient norm tensorboard histograms
        if tensorboard is not None:
            vis.pred_gradient_norm_histogram(self, train_loader, tensorboard, epoch, name='train-pred-grad')
            if val_loader is not None:
                vis.pred_gradient_norm_histogram(self, val_loader, tensorboard, epoch, name='val-pred-grad')

        # add gradient pair plots
        if train_loader.dataset.dataset_name == 'mnist':
            for p in [(0, 1), (4, 9)]:
                fig, _ = vis.pred_gradient_pair_scatter(self, train_loader, d1=p[0], d2=p[1])
                visualizations['gradients/train-pred-scatter-{}-{}'.format(p[0], p[1])] = fig
                if val_loader is not None:
                    fig, _ = vis.pred_gradient_pair_scatter(self, val_loader, d1=p[0], d2=p[1])
                    visualizations['gradients/val-pred-scatter-{}-{}'.format(p[0], p[1])] = fig

        return visualizations


class PredictGradOutputFixedForm(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network uses the form of output gradients.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, small_qtop=False,
                 sample_from_q=False, q_dist='Gaussian', **kwargs):
        super(PredictGradOutputFixedForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': 'lamb',
            'small_qtop': small_qtop,
            'sample_from_q': sample_from_q,
            'q_dist': q_dist,
            'class': 'PredictGradOutputFixedForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.small_qtop = small_qtop
        self.sample_from_q = sample_from_q
        self.q_dist = q_dist

        # TODO: fix everything
        if self.q_dist == 'Gaussian':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)))
        elif self.q_dist == 'Laplace':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)))
        else:
            raise NotImplementedError()

        # initialize the network
        self.classifier, output_shape = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                                    input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = output_shape[-1]

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        if small_qtop:
            self.q_top = torch.nn.Linear(q_base_shape[-1], self.num_classes).to(self.device)
        else:
            self.q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base_shape[-1], 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(self.device)

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
        pred = self.grad_replacement_class.apply(pred, grad_pred)

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
        if self.q_dist == 'Gaussian':
            info_penalty = self.lamb * losses.mse(y_one_hot, torch.softmax(q_label_pred, dim=1))
        elif self.q_dist == 'Laplace':
            info_penalty = self.lamb * losses.mad(y_one_hot, torch.softmax(q_label_pred, dim=1))
        else:
            raise NotImplementedError()

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty * \
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, info

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = super(PredictGradOutputFixedForm, self).visualize(train_loader, val_loader,
                                                                           tensorboard, epoch)

        # visualize q_label_pred
        fig, _ = vis.plot_predictions(self, train_loader, key='q_label_pred')
        visualizations['predictions/q-label-pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='q_label_pred')
            visualizations['predictions/q-label-pred-val'] = fig

        return visualizations


class PredictGradOutputFixedFormWithConfusion(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network uses the form of output gradients. A confusion matrix is also inferred.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, small_qtop=False,
                 sample_from_q=False, **kwargs):
        super(PredictGradOutputFixedFormWithConfusion, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': 'lamb',
            'small_qtop': small_qtop,
            'sample_from_q': sample_from_q,
            'class': 'PredictGradOutputFixedFormWithConfusion'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.small_qtop = small_qtop
        self.sample_from_q = sample_from_q
        self.grad_replacement_class = nn_utils.get_grad_replacement_class(
            sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)))

        # initialize the network
        self.classifier, output_shape = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                                    input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = output_shape[-1]

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        if small_qtop:
            self.q_top = torch.nn.Linear(q_base_shape[-1], self.num_classes).to(self.device)
        else:
            self.q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base_shape[-1], 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(self.device)

        # the confusion matrix trainable logits, (true, observed)
        Q_init = torch.zeros(size=(self.num_classes, self.num_classes), device=self.device, dtype=torch.float)
        Q_init += 1.0 * torch.eye(self.num_classes, device=self.device, dtype=torch.float)  # TODO: tune the constant
        self.Q_logits = torch.nn.Parameter(Q_init, requires_grad=True)

        # Q_init = 0.09 * torch.ones(size=(self.num_classes, self.num_classes), device=self.device, dtype=torch.float)
        # for i in range(self.num_classes):
        #     Q_init[i, i] += 0.1
        # self.Q_logits = torch.nn.Parameter(Q_init, requires_grad=False)

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
        pred = self.grad_replacement_class.apply(pred, grad_pred)

        # confusion matrix with probabilities
        Q = torch.softmax(self.Q_logits, dim=1)

        out = {
            'pred': pred,
            'q_label_pred': q_label_pred,
            'grad_pred': grad_pred,
            'Q': Q,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        q_label_pred = info['q_label_pred']
        grad_pred = info['grad_pred']
        Q = info['Q']
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        q_label_probs = torch.softmax(q_label_pred, dim=1)
        q_probs = torch.mm(q_label_probs, Q)
        q_prob = torch.sum(q_probs * y_one_hot, dim=1)  # select the probability corresponding to the observed y
        info_penalty = -self.lamb * torch.mean(torch.log(q_prob))

        # entropy minimization of q_label_pred
        # if there is no entropy minimization, it learns to set Q=Identity and make q_label_pred
        # close to uniform, which minimizes the H_{p,q} = <q_probs, y_one_hot> term.
        info_penalty -= 0.01 * torch.sum(q_label_probs * torch.log(q_label_probs), dim=1).mean()

        batch_losses = {
            'classifier': classifier_loss,
            'info_penalty': info_penalty
        }

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay *\
                           torch.mean(torch.sum(grad_pred**2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty * \
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, info

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = super(PredictGradOutputFixedFormWithConfusion, self).visualize(
            train_loader, val_loader, tensorboard, epoch)

        # visualize the confusion matrix
        if tensorboard is not None:
            fig, _ = vis.plot_confusion_matrix(torch.softmax(self.Q_logits, dim=1))
            visualizations['confusion-matrix'] = fig

        # visualize q_label_pred
        fig, _ = vis.plot_predictions(self, train_loader, key='q_label_pred')
        visualizations['predictions/q-label-pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='q_label_pred')
            visualizations['predictions/q-label-pred-val'] = fig

        return visualizations


class PredictGradOutputGeneralForm(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, small_qtop=False, **kwargs):
        super(PredictGradOutputGeneralForm, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'pretrained_arg': pretrained_arg,
            'device': device,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': 'lamb',
            'small_qtop': small_qtop,
            'class': 'PredictGradOutputGeneralForm'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.small_qtop = small_qtop

        # initialize the network
        self.classifier, _ = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                         input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        # NOTE: we want to use classifier parameters too
        # TODO: find a good parametrization
        if self.small_qtop:
            self.q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base_shape[-1] + self.num_classes, self.num_classes)).to(self.device)
            # TODO: add tanh here too, or delete below
        else:
            self.q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base_shape[-1] + self.num_classes, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes),
                torch.nn.Tanh()).to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred_before = self.classifier(x)
        pred_detached = pred_before.detach()

        # predict the gradient wrt to logits
        rx = self.q_base(x)
        grad_pred = self.q_top(torch.cat([rx, torch.softmax(pred_detached, dim=1)], dim=1))

        # change the gradients
        pred = nn_utils.GradReplacement.apply(pred_before, grad_pred)

        out = {
            'pred': pred,
            'pred_detached': pred_detached,
            'grad_pred': grad_pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        grad_pred = info['grad_pred']
        pred_detached = info['pred_detached']

        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=pred, target=y)

        # I(g : y | x) penalty
        grad_actual = torch.softmax(pred_detached, dim=1) - y_one_hot
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

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty * \
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, info


# TODO: pred_before needs to be detached
class PredictGradOutputGeneralFormUseLabel(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form and uses label information.
    """
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, **kwargs):
        super(PredictGradOutputGeneralFormUseLabel, self).__init__(**kwargs)

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'device': device,
            'pretrained_arg': pretrained_arg,
            'grad_weight_decay': grad_weight_decay,
            'grad_l1_penalty': grad_l1_penalty,
            'lamb': 'lamb',
            'class': 'PredictGradOutputGeneralFormUseLabel'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb

        # initialize the network
        self.classifier, _ = nn_utils.parse_feed_forward(args=self.architecture_args['classifier'],
                                                         input_shape=self.input_shape)
        self.classifier = self.classifier.to(self.device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_feed_forward(args=self.architecture_args['q-base'],
                                                                    input_shape=self.input_shape)
            self.q_base = self.q_base.to(self.device)

        # NOTE: we want to use classifier parameters too
        # TODO: find a good parametrization
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + 2 * self.num_classes, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(self.device)

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
        pred = nn_utils.GradReplacement.apply(pred_before, grad_pred)
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

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty * \
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, info
