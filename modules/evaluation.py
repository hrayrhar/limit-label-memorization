from tqdm import tqdm
import numpy as np


def compute_accuracy_with_bootstrapping(pred, target, n_iters=1000):
    """ Expects numpy arrays. pred should have shape (n_samples, n_classes), while
    target should have shape (n_samples,).
    """
    assert pred.shape[0] == target.shape[0]

    all_accuracies = []
    for _ in tqdm(range(n_iters), desc='bootstrapping') :
        indices = np.random.choice(pred.shape[0], size=pred.shape[0], replace=True)
        cur_pred = pred[indices]
        cur_target = target[indices]
        cur_accuracy = np.mean((cur_pred.argmax(axis=1) == cur_target).astype(np.float))
        all_accuracies.append(cur_accuracy)

    return {
        'mean': np.mean(all_accuracies),
        'std': np.std(all_accuracies)
    }
