from collections import defaultdict

from modules import visualization as vis
from nnlib.nnlib.method_utils import Method


class BaseClassifier(Method):
    """ Abstract class for classifiers.
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__()
        # initialize and use later
        self._current_iteration = defaultdict(lambda: 0)

    def on_iteration_end(self, partition, **kwargs):
        self._current_iteration[partition] += 1

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
