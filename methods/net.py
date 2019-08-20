from collections import defaultdict
from modules import nn, utils
import numpy as np
import torch


class NN(torch.nn.Module):
    """ Class for creating simple neural networks from architecture configs.
    """
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(NN, self).__init__()

        self.args = {
            'input_shape': input_shape,
            'architecture_args': architecture_args,
            'device': device,
            'class': 'NN'
        }

        assert len(input_shape) == 3
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args
        self.device = device

        # initialize and use later
        self._accuracy = defaultdict(list)

        # initialize the network
        self.network, _ = nn.parse_feed_forward(args=self.architecture_args['network'],
                                                input_shape=self.input_shape)
        self.network = self.network.to(self.device)

        print(self)

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        x = inputs[0].to(self.device)
        pred = self.network(x)

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info['pred']
        y = labels[0].to(self.device)

        loss = torch.nn.functional.cross_entropy(input=pred, target=y)

        batch_losses = {
            'loss': loss,
        }

        return batch_losses, info

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy[partition] = []

    def on_iteration_end(self, info, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(info['pred']).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)
        self._accuracy[partition].append((pred == batch_labels).astype(np.float).mean())

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy[partition])
        tensorboard.add_scalar('metrics/{}_accuracy'.format(partition), accuracy, epoch)
