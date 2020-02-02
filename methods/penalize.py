from modules import nn_utils, pretrained_models
import torch
import torch.nn.functional as F
from methods import BaseClassifier


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
            q_base = pretrained_models.get_pretrained_model(self.pretrained_arg, self.input_shape, self.device)

            # create the trainable part of the q_network
            q_top = torch.nn.Sequential(
                torch.nn.Linear(q_base.output_shape[-1], 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes)).to(self.device)

            self.q_network = torch.nn.Sequential(q_base, q_top)
        else:
            self.q_network, _ = nn_utils.parse_feed_forward(args=self.architecture_args['q-network'],
                                                            input_shape=self.input_shape)
            self.q_network = self.q_network.to(self.device)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        z = self.classifier_base(x)
        pred = self.classifier_last_layer(z)

        # predict labels from x
        q_label_pred = self.q_network(x)

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
