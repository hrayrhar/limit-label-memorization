""" Visualization routines to use for experiments.

These visualization tools will note save figures. That can be later done by
calling the savefig(fig, path) below. The purpose of this design is to make it
possible to use these tools in both jupyter notebooks and in ordinary scripts.
"""
import torch.nn.functional as F
from sklearn.manifold import TSNE
from modules import utils
from modules.data_utils import revert_normalization
import numpy as np
import os
import torch

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot


def get_image(x):
    """ Takes (1, H, W) or (3, H, W) and outputs (H, W, 3) """
    x = x.transpose((1, 2, 0))
    if x.shape[2] == 1:
        x = np.repeat(x, repeats=3, axis=2)
    return x


def reconstruction_plot(model, train_data, val_data, n_samples=5, plt=None):
    """Plots reconstruction examples for training & validation sets."""
    model.eval()
    if plt is None:
        plt = pyplot
    train_samples = [train_data[i][0] for i in range(n_samples)]
    val_samples = [val_data[i][0] for i in range(n_samples)]
    samples = torch.stack(train_samples + val_samples, dim=0)
    x_rec = model(inputs=[samples])['x_rec']
    x_rec = x_rec.reshape(samples.shape)
    samples = revert_normalization(samples, train_data)
    samples = utils.to_numpy(samples)
    x_rec = utils.to_numpy(x_rec)
    fig, ax = plt.subplots(nrows=2 * n_samples, ncols=2, figsize=(2, 2 * n_samples))
    for i in range(2 * n_samples):
        ax[i][0].imshow(get_image(samples[i]), vmin=0, vmax=1)
        ax[i][0].set_axis_off()
        ax[i][1].imshow(get_image(x_rec[i]), vmin=0, vmax=1)
        ax[i][1].set_axis_off()
    return fig, plt


def manifold_plot(model, example_shape, low=-1.0, high=+1.0, n_points=20, d1=0, d2=1, plt=None):
    """Plots reconstruction for varying dimensions d1 and d2, while the remaining dimensions are kept fixed."""
    model.eval()
    if plt is None:
        plt = pyplot
    image = np.zeros((example_shape[0], n_points * example_shape[1], n_points * example_shape[2]), dtype=np.float32)

    z = np.random.uniform(low=low, high=high, size=(model.hidden_shape[-1],))
    z1_grid = np.linspace(low, high, n_points)
    z2_grid = np.linspace(low, high, n_points)

    for i, z1 in enumerate(z1_grid):
        for j, z2 in enumerate(z2_grid):
            cur_z = np.copy(z)
            z[d1] = z1
            z[d2] = z2
            cur_z = cur_z.reshape((1, -1))
            x = utils.decode(model, cur_z).reshape(example_shape)
            x = utils.to_numpy(x)
            image[:, example_shape[1]*i: example_shape[1]*(i+1), example_shape[2]*j:example_shape[2]*(j+1)] = x
    fig, ax = plt.subplots(1, figsize=(10, 10))
    if image.shape[0] == 1:
        ax.imshow(image[0], vmin=0, vmax=1, cmap='gray')
    else:
        image = image.transpose((1, 2, 0))
        image = (255 * image).astype(np.uint8)
        ax.imshow(image)
    ax.axis('off')
    ax.set_ylabel('z_{}'.format(d1))
    ax.set_xlabel('z_{}'.format(d2))
    return fig, plt


def latent_scatter(model, data_loader, d1=0, d2=1, plt=None):
    """A scatter plot of latent factors on some 2-d subspace, with points colored according to test labels."""
    model.eval()
    if plt is None:
        plt = matplotlib.pyplot
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    z = []
    labels = []
    for batch_data, batch_labels in data_loader:
        z_batch = model(inputs=[batch_data])['z']
        z.append(utils.to_numpy(z_batch))
        labels.append(utils.to_numpy(batch_labels))
    z = np.concatenate(z, axis=0)
    labels = np.concatenate(labels, axis=0)
    fig, ax = plt.subplots(1)
    legend = []

    # if labels are not vectors, take the first coordiante
    if len(labels.shape) > 1:
        assert len(labels.shape) == 2
        labels = labels[:, 0]

    # replace labels with integers
    possible_labels = list(np.unique(labels))
    labels = [possible_labels.index(label) for label in labels]

    # select up to 10000 points
    cnt = min(10000, len(labels))
    indices = np.random.choice(len(labels), cnt)
    z = z[indices]
    labels = [labels[idx] for idx in indices]

    for i in np.unique(labels):
        indices = (labels == i)
        ax.scatter(z[indices, d1], z[indices, d2], marker='.', color=tab[i], alpha=0.5, edgecolor='', label=i)
        legend.append(str(i))
    fig.legend(legend)
    ax.set_xlabel("$Z_{}$".format(d1))
    ax.set_ylabel("$Z_{}$".format(d2))
    L = np.percentile(z, q=5, axis=0)
    R = np.percentile(z, q=95, axis=0)
    ax.set_xlim(L[d1], R[d1])
    ax.set_ylim(L[d2], R[d2])
    ax.set_title('Latent space')
    return fig, plt


def latent_space_tsne(model, data_loader, plt=None):
    """A scatter plot of latent factors on some 2-d subspace, with points colored according to test labels."""
    model.eval()
    if plt is None:
        plt = matplotlib.pyplot
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    z = []
    labels = []
    for batch_data, batch_labels in data_loader:
        z_batch = model(inputs=[batch_data])['z']
        z.append(utils.to_numpy(z_batch))
        labels.append(utils.to_numpy(batch_labels))
    z = np.concatenate(z, axis=0)
    labels = np.concatenate(labels, axis=0)
    fig, ax = plt.subplots(1)
    legend = []

    # if labels are not vectors, take the first coordinate
    if len(labels.shape) > 1:
        assert len(labels.shape) == 2
        labels = labels[:, 0]

    # replace labels with integers
    possible_labels = list(np.unique(labels))
    labels = [possible_labels.index(label) for label in labels]

    # run TSNE for embedding z into 2 dimensional space
    z = TSNE(n_components=2).fit_transform(z)

    # select up to 10000 points
    cnt = min(10000, len(labels))
    indices = np.random.choice(len(labels), cnt)
    z = z[indices]
    labels = [labels[idx] for idx in indices]

    for i in np.unique(labels):
        indices = (labels == i)
        ax.scatter(z[indices, 0], z[indices, 1], marker='.', color=tab[i], alpha=0.5, edgecolor='', label=i)
        legend.append(str(i))
    fig.legend(legend)
    L = np.percentile(z, q=5, axis=0)
    R = np.percentile(z, q=95, axis=0)
    ax.set_xlim(L[0], R[0])
    ax.set_ylim(L[1], R[1])
    ax.set_title('Latent space TSNE plot')
    return fig, plt


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


def plot_predictions(model, data_loader, key, plt=None):
    if plt is None:
        plt = matplotlib.pyplot
    model.eval()

    n_examples = 10
    pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                  output_keys_regexp=key,
                                  max_num_examples=n_examples,
                                  description='plot_predictions:{}'.format(key))[key]
    probs = torch.softmax(pred, dim=1)
    probs = utils.to_numpy(probs)

    data = [data_loader.dataset[i][0] for i in range(n_examples)]
    labels = [data_loader.dataset[i][1] for i in range(n_examples)]
    samples = torch.stack(data, dim=0)
    samples = revert_normalization(samples, data_loader.dataset)
    samples = utils.to_numpy(samples)

    fig, ax = plt.subplots(nrows=n_examples, ncols=2, figsize=(2*2, 2*n_examples))
    for i in range(n_examples):
        ax[i][0].imshow(get_image(samples[i]), vmin=0, vmax=1)
        ax[i][0].set_axis_off()
        ax[i][0].set_title('labels as {}'.format(labels[i]))

        ax[i][1].bar(range(model.num_classes), probs[i])
        ax[i][1].set_xticks(range(model.num_classes))

    return fig, plt


def savefig(fig, path):
    dir_name = os.path.dirname(path)
    if dir_name != '':
        utils.make_path(dir_name)
    fig.savefig(path)
