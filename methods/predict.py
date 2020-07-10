import numpy as np
import torch
import torch.nn.functional as F

from modules import nn_utils, losses, pretrained_models
from modules import visualization as vis
from methods import BaseClassifier
from nnlib.nnlib import utils


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


class PredictGradOutput(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    """
    @utils.capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, sample_from_q=False,
                 q_dist='Gaussian', loss_function='ce', detach=True, load_from=None,
                 warm_up=0, **kwargs):
        super(PredictGradOutput, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.sample_from_q = sample_from_q
        self.q_dist = q_dist
        self.detach = detach
        self.loss_function = loss_function
        self.load_from = load_from
        self.warm_up = warm_up

        # lamb is the coefficient in front of the H(p,q) term. It controls the variance of predicted gradients.
        if self.q_dist == 'Gaussian':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)), q_dist=self.q_dist)
        elif self.q_dist == 'Laplace':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(2.0) / (self.lamb + 1e-6), q_dist=self.q_dist)
        elif self.q_dist in ['dot', 'ce']:
            assert not self.sample_from_q
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(sample=False)
        else:
            raise NotImplementedError()

        # initialize the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape)
        self.classifier = self.classifier.to(device)
        self.num_classes = output_shape[-1]

        if self.pretrained_arg is not None:
            q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, device)

            # create the trainable part of the q_network
            q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base.output_shape[-1], 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(device)

            self.q_network = torch.nn.Sequential(q_base, q_top)
            self.q_network = self.q_network.to(device)
        else:
            self.q_network, _ = nn_utils.parse_network_from_config(args=self.architecture_args['q-network'],
                                                                   input_shape=self.input_shape)
            self.q_network = self.q_network.to(device)

            if self.load_from is not None:
                print("Loading the gradient predictor model from {}".format(load_from))
                import methods
                stored_net = utils.load(load_from, methods=methods, device='cpu')
                stored_net_params = dict(stored_net.classifier.named_parameters())
                for key, param in self.q_network.named_parameters():
                    param.data = stored_net_params[key].data.to(device)

        self.q_loss = None
        if self.loss_function == 'none':  # predicted gradient has general form
            self.q_loss = torch.nn.Sequential(
                torch.nn.Linear(2 * self.num_classes, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(device)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred = self.classifier(x)

        # predict the gradient wrt to logits
        q_label_pred = self.q_network(x)
        q_label_pred_softmax = torch.softmax(q_label_pred, dim=1)
        if self.detach:
            # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
            pred_softmax = torch.softmax(pred, dim=1).detach()
        else:
            pred_softmax = torch.softmax(pred, dim=1)
        if self.loss_function == 'ce':
            grad_pred = pred_softmax - q_label_pred_softmax
        elif self.loss_function == 'mae':
            grad_pred = torch.sum(q_label_pred_softmax * pred_softmax, dim=1).unsqueeze(dim=-1) *\
                        (pred_softmax - q_label_pred_softmax)
        elif self.loss_function == 'none':
            grad_pred = self.q_loss(torch.cat([pred_softmax, q_label_pred_softmax], dim=1))
        else:
            raise NotImplementedError()

        # replace the gradients
        pred_before = pred
        pred = self.grad_replacement_class.apply(pred, grad_pred)

        out = {
            'pred': pred,
            'q_label_pred': q_label_pred,
            'grad_pred': grad_pred,
            'pred_before': pred_before
        }

        return out

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        pred_before = outputs['pred_before']
        grad_pred = outputs['grad_pred']
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=outputs['pred'], target=y)

        # compute grad actual
        if self.detach:
            # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
            pred_softmax = torch.softmax(pred_before.detach(), dim=1)
        else:
            pred_softmax = torch.softmax(pred_before, dim=1)
        if self.loss_function in ['ce', 'none']:
            grad_actual = pred_softmax - y_one_hot
        elif self.loss_function == 'mae':
            grad_actual = torch.sum(pred_softmax * y_one_hot, dim=1).unsqueeze(dim=-1) *\
                          (pred_softmax - y_one_hot)
        else:
            raise NotImplementedError()

        # I(g : y | x) penalty
        if self.q_dist == 'Gaussian':
            info_penalty = losses.mse(grad_pred, grad_actual)
        elif self.q_dist == 'Laplace':
            info_penalty = losses.mae(grad_pred, grad_actual)
        elif self.q_dist == 'dot':
            # this corresponds to Taylor approximation of L(w + g_t)
            info_penalty = -torch.mean((grad_pred * grad_actual).sum(dim=1), dim=0)
        elif self.q_dist == 'ce':
            # TODO: clarify which distribution will give this
            info_penalty = losses.get_classification_loss(target=y_one_hot,
                                                          pred=outputs['q_label_pred'],
                                                          loss_function='ce')
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
            grad_l1_loss = self.grad_l1_penalty *\
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, outputs

    def on_epoch_start(self, partition, epoch, **kwargs):
        super(PredictGradOutput, self).on_epoch_start(partition=partition, epoch=epoch, **kwargs)
        if partition == 'train':
            requires_grad = (epoch >= self.warm_up)
            for param in self.classifier.parameters():
                param.requires_grad = requires_grad

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = super(PredictGradOutput, self).visualize(train_loader, val_loader,
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
    @utils.capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, small_qtop=False,
                 sample_from_q=False, **kwargs):
        super(PredictGradOutputFixedFormWithConfusion, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.small_qtop = small_qtop
        self.sample_from_q = sample_from_q
        self.grad_replacement_class = nn_utils.get_grad_replacement_class(
            sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)))

        # initialize the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape)
        self.classifier = self.classifier.to(device)
        self.num_classes = output_shape[-1]

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_network_from_config(args=self.architecture_args['q-base'],
                                                                           input_shape=self.input_shape)
            self.q_base = self.q_base.to(device)

        if small_qtop:
            self.q_top = torch.nn.Linear(q_base_shape[-1], self.num_classes).to(device)
        else:
            self.q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base_shape[-1], 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(device)

        # the confusion matrix trainable logits, (true, observed)
        Q_init = torch.zeros(size=(self.num_classes, self.num_classes), device=device, dtype=torch.float)
        Q_init += 1.0 * torch.eye(self.num_classes, device=device, dtype=torch.float)  # TODO: tune the constant
        self.Q_logits = torch.nn.Parameter(Q_init, requires_grad=True)

        # Q_init = 0.09 * torch.ones(size=(self.num_classes, self.num_classes), device=device, dtype=torch.float)
        # for i in range(self.num_classes):
        #     Q_init[i, i] += 0.1
        # self.Q_logits = torch.nn.Parameter(Q_init, requires_grad=False)

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

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        pred = outputs['pred']
        q_label_pred = outputs['q_label_pred']
        grad_pred = outputs['grad_pred']
        Q = outputs['Q']
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

        return batch_losses, outputs

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


# TODO: pred_before needs to be detached
class PredictGradOutputGeneralFormUseLabel(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    The q network has general form and uses label information.
    """
    @utils.capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, pretrained_arg=None, device='cuda',
                 grad_weight_decay=0.0, grad_l1_penalty=0.0, lamb=1.0, **kwargs):
        super(PredictGradOutputGeneralFormUseLabel, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.pretrained_arg = pretrained_arg
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb

        # initialize the network
        self.classifier, _ = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                input_shape=self.input_shape)
        self.classifier = self.classifier.to(device)
        self.num_classes = self.architecture_args['classifier'][-1]['dim']

        if self.pretrained_arg is not None:
            self.q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, device)
            q_base_shape = self.q_base.output_shape
        else:
            self.q_base, q_base_shape = nn_utils.parse_network_from_config(args=self.architecture_args['q-base'],
                                                                           input_shape=self.input_shape)
            self.q_base = self.q_base.to(device)

        # NOTE: we want to use classifier parameters too
        # TODO: find a good parametrization
        self.q_top = torch.nn.Sequential(
            torch.nn.Linear(q_base_shape[-1] + 2 * self.num_classes, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, self.num_classes)).to(device)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred_before = self.classifier(x)

        out = {
            'pred_before': pred_before,
        }

        return out

    def compute_loss(self, inputs, labels, outputs, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        pred_before = outputs['pred_before']

        x = inputs[0].to(self.device)
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # predict the gradient wrt to logits
        rx = self.q_base(x)
        grad_pred = self.q_top(torch.cat([rx, pred_before, y_one_hot], dim=1))
        outputs['grad_pred'] = grad_pred

        # change the gradients
        pred = nn_utils.GradReplacement.apply(pred_before, grad_pred)
        outputs['pred'] = pred

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
            grad_l1_loss = self.grad_l1_penalty *\
                           torch.mean(torch.sum(torch.abs(grad_pred), dim=1), dim=0)
            batch_losses['pred_grad_l1'] = grad_l1_loss

        return batch_losses, outputs
