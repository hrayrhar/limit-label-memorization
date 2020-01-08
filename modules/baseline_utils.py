from modules import utils
import torch
import numpy as np


def estimate_transition(load_from, data_loader, device='cpu', batch_size=256):
    """ Estimates the label noise matrix. The code is adapted form the original implementation.
    Source: https://github.com/giorgiop/loss-correction/.
    """
    assert load_from is not None
    model = utils.load(load_from, device=device)
    pred = utils.apply_on_dataset(model=model, dataset=data_loader.dataset,
                                  batch_size=batch_size, cpu=True, description="Estimating transition matrix",
                                  output_keys_regexp='pred')['pred']
    pred = torch.softmax(pred, dim=1)
    pred = utils.to_numpy(pred)

    c = model.num_classes
    T = np.zeros((c, c))
    filter_outlier = True

    # find a 'perfect example' for each class
    for i in range(c):
        if not filter_outlier:
            idx_best = np.argmax(pred[:, i])
        else:
            thresh = np.percentile(pred[:, i], 97, interpolation='higher')
            robust_eta = pred[:, i]
            robust_eta[robust_eta >= thresh] = 0.0
            idx_best = np.argmax(robust_eta)

        for j in range(c):
            T[i, j] = pred[idx_best, j]

    # row normalize
    row_sums = T.sum(axis=1, keepdims=True)
    T /= row_sums

    T = torch.tensor(T, dtype=torch.float).to(device)
    print(T)

    return T
