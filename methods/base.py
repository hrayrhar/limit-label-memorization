from collections import defaultdict
from modules import utils
from modules import visualization as vis
import numpy as np
import torch


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

    def before_weight_update(self, **kwargs):
        pass

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
