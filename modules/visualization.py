""" Visualization routines to use for experiments.

These visualization tools will note save figures. That can be later done by
calling the savefig(fig, path) below. The purpose of this design is to make it
possible to use these tools in both jupyter notebooks and in ordinary scripts.
"""
import torch.nn.functional as F
import torch

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

import nnlib.nnlib.visualizations
from nnlib.nnlib import utils


# import some nnlib visualizations
reconstruction_plot = nnlib.nnlib.visualizations.reconstruction_plot
manifold_plot = nnlib.nnlib.visualizations.manifold_plot
latent_scatter = nnlib.nnlib.visualizations.latent_scatter
latent_space_tsne = nnlib.nnlib.visualizations.latent_space_tsne
plot_predictions = nnlib.nnlib.visualizations.plot_predictions

# import some utils from nnlib visualizations
get_image = nnlib.nnlib.visualizations.get_image
savefig = nnlib.nnlib.visualizations.savefig


def ce_gradient_norm_histogram(model, data_loader, tensorboard, epoch, name, max_num_examples=5000):
    model.eval()

    pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                  output_keys_regexp='pred', description='grad-histogram:pred',
                                  max_num_examples=max_num_examples)['pred']
    n_examples = min(len(data_loader.dataset), max_num_examples)
    labels = []
    for idx in range(n_examples):
        labels.append(data_loader.dataset[idx][1])
    labels = torch.tensor(labels, dtype=torch.long)
    labels = F.one_hot(labels, num_classes=model.num_classes).float()
    labels = utils.to_cpu(labels)

    grad_wrt_logits = torch.softmax(pred, dim=-1) - labels
    grad_norms = torch.sum(grad_wrt_logits**2, dim=-1)
    grad_norms = utils.to_numpy(grad_norms)

    try:
        tensorboard.add_histogram(tag=name, values=grad_norms, global_step=epoch)
    except ValueError as e:
        print("Tensorboard histogram error: {}".format(e))


def ce_gradient_pair_scatter(model, data_loader, d1=0, d2=1, max_num_examples=2000, plt=None):
    if plt is None:
        plt = matplotlib.pyplot
    model.eval()

    pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                  output_keys_regexp='pred',
                                  max_num_examples=max_num_examples,
                                  description='grad-pair-scatter:pred')['pred']
    n_examples = min(len(data_loader.dataset), max_num_examples)
    labels = []
    for idx in range(n_examples):
        labels.append(data_loader.dataset[idx][1])
    labels = torch.tensor(labels, dtype=torch.long)
    labels = F.one_hot(labels, num_classes=model.num_classes).float()
    labels = utils.to_cpu(labels)
    grad_wrt_logits = torch.softmax(pred, dim=-1) - labels
    grad_wrt_logits = utils.to_numpy(grad_wrt_logits)

    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.scatter(grad_wrt_logits[:, d1], grad_wrt_logits[:, d2])
    ax.set_xlabel(str(d1))
    ax.set_ylabel(str(d2))
    # L = np.percentile(grad_wrt_logits, q=5, axis=0)
    # R = np.percentile(grad_wrt_logits, q=95, axis=0)
    # ax.set_xlim(L[d1], R[d1])
    # ax.set_ylim(L[d2], R[d2])
    ax.set_title('Two coordinates of grad wrt to logits')
    return fig, plt


def pred_gradient_norm_histogram(model, data_loader, tensorboard, epoch, name, max_num_examples=5000):
    model.eval()
    grad_pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                       output_keys_regexp='grad_pred', description='grad-histogram:grad_pred',
                                       max_num_examples=max_num_examples)['grad_pred']
    grad_norms = torch.sum(grad_pred**2, dim=-1)
    grad_norms = utils.to_numpy(grad_norms)

    try:
        tensorboard.add_histogram(tag=name, values=grad_norms, global_step=epoch)
    except ValueError as e:
        print("Tensorboard histogram error: {}".format(e))


def pred_gradient_pair_scatter(model, data_loader, d1=0, d2=1, max_num_examples=2000, plt=None):
    if plt is None:
        plt = matplotlib.pyplot
    model.eval()
    grad_pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                       output_keys_regexp='grad_pred',
                                       max_num_examples=max_num_examples,
                                       description='grad-pair-scatter:grad_pred')['grad_pred']
    grad_pred = utils.to_numpy(grad_pred)
    fig, ax = plt.subplots(1, figsize=(5, 5))
    plt.scatter(grad_pred[:, d1], grad_pred[:, d2])
    ax.set_xlabel(str(d1))
    ax.set_ylabel(str(d2))
    # L = np.percentile(grad_pred, q=5, axis=0)
    # R = np.percentile(grad_pred, q=95, axis=0)
    # ax.set_xlim(L[d1], R[d1])
    # ax.set_ylim(L[d2], R[d2])
    ax.set_title('Two coordinates of grad wrt to logits')
    return fig, plt


def plot_confusion_matrix(Q, plt=None):
    if plt is None:
        plt = matplotlib.pyplot
    num_classes = Q.shape[0]
    fig, ax = plt.subplots(1, figsize=(5, 5))
    im = ax.imshow(utils.to_numpy(Q))
    fig.colorbar(im)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('observed')
    ax.set_ylabel('true')
    return fig, plt
