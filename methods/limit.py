import numpy as np
import torch
import torch.nn.functional as F

from nnlib.nnlib import visualizations as vis
from nnlib.nnlib import losses, utils
from modules import nn_utils
from methods.predict import PredictGradBaseClassifier


class LIMIT(PredictGradBaseClassifier):
    """ The main method of "Improving generalization by controlling label-noise
    information in neural network weights" paper. This method trains a classifier
    using gradients predict by another network without directly using labels.
    As in the paper, only the gradient with respect to the output of the last layer
    is predicted, the remaining gradients are computed using backpropogation, starting
    with the predicted gradient.

    For more details, refer to the paper at https://arxiv.org/abs/2002.07933.
    """
    @utils.capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda',
                 grad_weight_decay=0.0, lamb=1.0, sample_from_q=False,
                 q_dist='Gaussian', load_from=None, warm_up=0, **kwargs):
        """
        :param input_shape: the input shape of an example. E.g. for CIFAR-10 this is (3, 32, 32).
        :param architecture_args: dictionary usually parsed from a json file from the `configs`
            directory. This specifies the architecture of the classifier and the architecture of
            the gradient predictor network: `q-network`. If you don't want to parse the networks
            from arguments, you can modify the code so that self.classifier and self.q_network
            directly point to the correct models.
        :param device: the device on which the model is stored and executed.
        :param grad_weight_decay: the strength of regularization of the mean of the predicted gradients,
            ||\mu||_2^2. Usually values from [0.03 - 10] work well. Refer to the paper for more guidance.
        :param lamb: this is the coefficient in front of the H(p, q) term. Unless `sample_from_q=True`,
            setting this to anything but 1.0 has no effect. When `sample_from_q=True`, `lamb` specifies
            the variance of the predicted gradients.
        :param sample_from_q: whether to sample from the q distribution (predicted gradient distribution),
            or to use the mean.
        :param q_dist: what distribution predicted gradients should follow. Options are 'Gaussian', 'Laplace',
            'ce'. The names of the first 2 speak about themselves and were used in the paper under names LIMIT_G,
            and LIMIT_L. The option 'ce' corresponds to a hypothetical case when H(p,q) reduces to
            CE(q_label_pred, actual_label). This latter option may work better for some datasets. When
            `q_dist=ce`, then `sample_from_q` has to be false.
        :param load_from: path to a file where another model (already trained) was saved. This will be loaded,
            and the training will continue from this starting point. Note that the saved model needs to be saved
            using nnlib.nnlib.utils.save function.
        :param warm_up: number of initial epochs for which the classifier is not trained at all. This is done to
            give the q-network enough time to learn meaningful gradient predictions before using those predicted
            gradients to train the classifier.
        :param kwargs: additional keyword arguments that are passed to the parent methods. For this class it
            can be always empty.
        """
        super(LIMIT, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.grad_weight_decay = grad_weight_decay
        self.lamb = lamb
        self.sample_from_q = sample_from_q
        self.q_dist = q_dist
        self.load_from = load_from
        self.warm_up = warm_up

        # lamb is the coefficient in front of the H(p,q) term. It controls the variance of predicted gradients.
        if self.q_dist == 'Gaussian':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)), q_dist=self.q_dist)
        elif self.q_dist == 'Laplace':
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q, standard_dev=np.sqrt(2.0) / (self.lamb + 1e-6), q_dist=self.q_dist)
        elif self.q_dist == 'ce':
            # This is not an actual distributions. Instead, this correspond to hypothetical case when
            # H(p,q) term results to ce(q_label_pred, actual_label).
            assert not self.sample_from_q
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(sample=False)
        else:
            raise NotImplementedError()

        # initialize the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape)
        self.classifier = self.classifier.to(device)
        self.num_classes = output_shape[-1]

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

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        x = inputs[0].to(self.device)

        # compute classifier predictions
        pred = self.classifier(x)

        # predict the gradient wrt to logits
        q_label_pred = self.q_network(x)
        q_label_pred_softmax = torch.softmax(q_label_pred, dim=1)
        # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
        pred_softmax = torch.softmax(pred, dim=1).detach()
        grad_pred = pred_softmax - q_label_pred_softmax

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

        # compute the actual gradient
        # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
        pred_softmax = torch.softmax(pred_before.detach(), dim=1)
        grad_actual = pred_softmax - y_one_hot

        # I(g : y | x) penalty
        if self.q_dist == 'Gaussian':
            info_penalty = losses.mse(grad_pred, grad_actual)
        elif self.q_dist == 'Laplace':
            info_penalty = losses.mae(grad_pred, grad_actual)
        elif self.q_dist == 'ce':
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
            grad_l2_loss = self.grad_weight_decay * \
                           torch.mean(torch.sum(grad_pred ** 2, dim=1), dim=0)
            batch_losses['pred_grad_l2'] = grad_l2_loss

        return batch_losses, outputs

    def on_epoch_start(self, partition, epoch, **kwargs):
        super(LIMIT, self).on_epoch_start(partition=partition, epoch=epoch, **kwargs)
        if partition == 'train':
            requires_grad = (epoch >= self.warm_up)
            for param in self.classifier.parameters():
                param.requires_grad = requires_grad

    def visualize(self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs):
        visualizations = super(LIMIT, self).visualize(train_loader, val_loader,
                                                      tensorboard, epoch)

        # visualize q_label_pred
        fig, _ = vis.plot_predictions(self, train_loader, key='q_label_pred')
        visualizations['predictions/q-label-pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='q_label_pred')
            visualizations['predictions/q-label-pred-val'] = fig

        return visualizations
