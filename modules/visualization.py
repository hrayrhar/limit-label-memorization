""" Visualization routines to use for experiments.

These visualization tools will note save figures. That can be later done by
calling the savefig(fig, path) below. The purpose of this design is to make it
possible to use these tools in both jupyter notebooks and in ordinary scripts.
"""
from modules import utils
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
    train_data = [train_data[i][0] for i in range(n_samples)]
    val_data = [val_data[i][0] for i in range(n_samples)]
    samples = torch.stack(train_data + val_data, dim=0)
    x_rec = model(inputs=[samples])['x_rec']
    x_rec = x_rec.reshape(samples.shape)
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


def factors_vs_latents(model, dataset, plt=None):
    model.eval()
    if plt is None:
        plt = matplotlib.pyplot
    tab = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
           'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    fs = [f for x, f in dataset]
    fs = torch.stack(fs, dim=0)
    fs = fs.numpy()

    zs = utils.apply_on_dataset(model=model, dataset=dataset, output_keys_regexp='z')
    zs = zs['z'].numpy()

    cnt = min(10000, len(fs))
    indices = np.random.choice(len(fs), cnt)
    fs = fs[indices]
    zs = zs[indices]

    n_factors = fs.shape[1]
    n_latents = zs.shape[1]

    fig, ax = plt.subplots(nrows=n_factors, ncols=n_latents, figsize=(2 * n_latents, 2 * n_factors))
    fig.suptitle('X: latents, Y: factors')
    for i in range(n_factors):
        for j in range(n_latents):
            if len(np.unique(fs[:, i])) >= 10:
                heatmap, _, _ = np.histogram2d(zs[:, j], fs[:, i], bins=50)
                ax[i][j].imshow(heatmap.T, origin='lower', cmap='Oranges')
            else:
                vals = np.unique(fs[:, i])
                for idx, f in enumerate(vals):
                    indices_mask = (fs[:, i] == f)
                    hs, xs, = np.histogram(zs[indices_mask][:, j], bins=50)
                    ax[i][j].fill_between((xs[:-1] + xs[1:]) / 2.0, hs, color=tab[idx], alpha=1.0 / len(vals))

    return fig, plt


def generated_path_vs_target_path_plot(model, loader, n_plots=8, plt=None):
    model.eval()
    if plt is None:
        plt = matplotlib.pyplot

    xs = []
    zs = []
    N = len(loader.dataset)
    for i in range(n_plots):
        data, _ = loader.dataset[np.random.randint(N)]
        xs.append(utils.to_tensor(data[0]))
        zs.append(utils.to_tensor(data[1]))

    xs = torch.stack(xs)
    zs = torch.stack(zs)
    ms = torch.zeros((xs.shape[0], xs.shape[1] - 1))
    sequence = model.forward(inputs=[xs, zs, ms])['sequence']

    n_cols = sequence.shape[1]
    fig, ax = plt.subplots(nrows=2*n_plots, ncols=n_cols, figsize=(2 * n_cols, 2 * 2 * n_plots))
    for i in range(n_plots):
        for j in range(sequence.shape[1]):
            ax[2 * i][j].imshow(get_image(utils.to_numpy(xs[i, j])), vmin=0, vmax=1)
            ax[2 * i][j].set_axis_off()
            ax[2 * i + 1][j].imshow(get_image(utils.to_numpy(sequence[i, j])), vmin=0, vmax=1)
            ax[2 * i + 1][j].set_axis_off()

    return fig, plt


def coordinate_change(model, dataset, n_plots=8, concat=True, plt=None):
    model.eval()
    if plt is None:
        plt = matplotlib.pyplot

    samples = [dataset[i][0] for i in range(n_plots)]
    samples = torch.stack(samples, dim=0)
    info = model.forward([samples])
    x_rec = info['x_rec']
    x_rec_changed = info['x_rec_changed']
    targets = info['targets']
    z = info['z']
    z_changed = info['z_changed']

    if concat:
        classifier_prediction = model.classifier(torch.cat([x_rec, x_rec_changed], dim=1))
    else:
        classifier_prediction = model.classifier(x_rec - x_rec_changed)

    fig, ax = plt.subplots(nrows=n_plots, ncols=5, figsize=(2*5, 2*n_plots))
    for i in range(n_plots):
        if model.loss_type == 'classification':
            cur_target = targets[i]
        else:
            cur_target = torch.argmax(targets[i])

        ax[i][0].imshow(get_image(utils.to_numpy(samples[i])), vmin=0, vmax=1)
        ax[i][0].set_axis_off()
        ax[i][0].set_title('source')
        ax[i][1].imshow(get_image(utils.to_numpy(x_rec[i])), vmin=0, vmax=1)
        ax[i][1].set_axis_off()
        ax[i][1].set_title('z_{} = {:.2f}'.format(cur_target, z[i, cur_target]))
        ax[i][2].imshow(get_image(utils.to_numpy(x_rec_changed[i])), vmin=0, vmax=1)
        ax[i][2].set_axis_off()
        ax[i][2].set_title('z_{} = {:.2f}'.format(cur_target, z_changed[i, cur_target]))
        ax[i][3].imshow(get_image(utils.to_numpy(torch.abs(x_rec[i] - x_rec_changed[i]))), vmin=0, vmax=1)
        ax[i][3].set_axis_off()
        ax[i][3].set_title('|x1 - x2|')

        classifier_probabilities = utils.to_numpy(torch.softmax(classifier_prediction[i], dim=0))
        bar_color = ('green' if np.argmax(classifier_probabilities) == cur_target else 'red')
        ax[i][4].bar(range(z.shape[1]), classifier_probabilities, color=bar_color)
        ax[i][4].set_axis_off()
        ax[i][4].set_title('classifier')

    return fig, plt


def savefig(fig, path):
    dir_name = os.path.dirname(path)
    if dir_name != '':
        utils.make_path(dir_name)
    fig.savefig(path)
