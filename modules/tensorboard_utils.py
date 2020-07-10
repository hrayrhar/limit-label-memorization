from scipy.ndimage.filters import gaussian_filter1d
import tensorflow as tf
import os


def find_latest_tfevents_file(dir_path):
    files = os.listdir(dir_path)
    files = list(filter(lambda x: x.find("tfevents") != -1, files))
    if len(files) == 0:
        return None
    return sorted(files)[-1]


def extract_tag_from_tensorboard(tb_events_file, tag):
    ret_events = []
    for event in tf.compat.v1.train.summary_iterator(tb_events_file):
        step = event.step
        for value in event.summary.value:
            if value.tag == tag:
                ret_events.append((step, value.simple_value))
    return ret_events


def plot_from_tensorboard(plt, files, common_prefix=None, savename=None, xmin=None, xmax=None, ymin=None, ymax=None,
                          legend_args=None, xlabel='Epochs', ylabel='Accuracy', title=None, smoothing=False,
                          tag='metrics/train_accuracy', figsize=(9, 4), xticks=None, yticks=None,
                          linewidth=None):
    """ Plots a given tag from multiple tensorboard log files.
    :param files: list of dictionaries, that have keys 'name', 'dir_path', ['marker'], ['markevery']
    """
    fig, ax = plt.subplots(figsize=figsize)
    for d in files:
        dir_path = d['dir_path']
        if common_prefix is not None:
            dir_path = os.path.join(common_prefix, dir_path)

        tfevents_file = find_latest_tfevents_file(dir_path)
        tfevents_file = os.path.join(dir_path, tfevents_file)
        events = extract_tag_from_tensorboard(tfevents_file, tag)
        steps = [p[0] for p in events]
        values = [p[1] for p in events]
        if smoothing:
            values = gaussian_filter1d(values, sigma=2)
        ax.plot(steps, values, label=d['name'], color=d.get('color', None),
                linestyle=d.get('linestyle', '-'), marker=d.get('marker', None),
                markevery=d.get('markevery', None), linewidth=linewidth)

    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    if legend_args is not None:
        ax.legend(**legend_args)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    return fig, ax
