from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import os

# for fixing RuntimeError: received 0 items of ancdata
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def split(dataset, val_ratio, seed):
    train_cnt = int((1 - val_ratio) * len(dataset))
    np.random.seed(seed)
    perm = np.random.permutation(len(dataset))
    train_indices = perm[:train_cnt]
    val_indices = perm[train_cnt:]
    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    return train_data, val_data


def remove_random_chunks(x, prob):
    """ Divide the image into 4x4 patches and remove patches randomly.
    """
    ret = x.clone()
    chunk_size = 4
    assert x.shape[0] % chunk_size == 0
    assert x.shape[1] % chunk_size == 0
    n_x_blocks = x.shape[0] // chunk_size
    n_y_blocks = x.shape[1] // chunk_size
    random_value = np.random.uniform(size=(n_x_blocks, n_y_blocks))
    for i in range(n_x_blocks):
        for j in range(n_y_blocks):
            if random_value[i, j] < prob:
                ret[i * chunk_size:(i+1) * chunk_size, j * chunk_size:(j+1) * chunk_size] = 0
    return ret


def create_remove_random_chunks_function(prob=0.5):
    """ Returns a remove_random_chunks function with given probability.
    """
    def modify(x):
        return remove_random_chunks(x, prob=prob)
    return modify


def load_mnist_datasets(val_ratio=0.2, noise_level=0.0, transform_function=None,
                        transform_validation=False, skip_normalization=False, seed=42):
    data_dir = os.path.join(os.path.dirname(__file__), '../data/mnist/')

    all_transforms = [transforms.ToTensor()]
    if not skip_normalization:
        # NOTE: this normalization is set so that models pretrained on ImageNet work well
        all_transforms.append(transforms.Normalize(mean=[0.456], std=[0.224]))
    composed_transform = transforms.Compose(all_transforms)

    train_data = datasets.MNIST(data_dir, download=True, train=True, transform=composed_transform)
    test_data = datasets.MNIST(data_dir, download=True, train=False, transform=composed_transform)
    train_data, val_data = split(train_data, val_ratio, seed)

    train_data.dataset_name = 'mnist'
    val_data.dataset_name = 'mnist'
    test_data.dataset_name = 'mnist'

    # corrupt noise_level percent of the training labels
    is_corrupted = np.zeros(len(train_data), dtype=int)  # 0 clean, 1 corrupted, 2 accidentally correct
    for current_idx, sample_idx in enumerate(train_data.indices):
        if np.random.uniform(0, 1) < noise_level:
            new_label = np.random.randint(10)
            if new_label == train_data.dataset.targets[sample_idx]:
                is_corrupted[current_idx] = 2
            else:
                is_corrupted[current_idx] = 1
            train_data.dataset.targets[sample_idx] = new_label

    # modify images if needed
    if transform_function is not None:
        # transform training samples
        for sample_idx in train_data.indices:
            train_data.dataset.data[sample_idx] = transform_function(train_data.dataset.data[sample_idx])

        if transform_validation:
            # transform validation samples
            for sample_idx in val_data.indices:
                val_data.dataset.data[sample_idx] = transform_function(val_data.dataset.data[sample_idx])

            # transform testing samples
            for sample_idx in range(len(test_data)):
                test_data.data[sample_idx] = transform_function(test_data.data[sample_idx])

    return train_data, val_data, test_data, is_corrupted


def load_mnist_loaders(val_ratio=0.2, batch_size=128, noise_level=0.0, seed=42,
                       drop_last=False, num_train_examples=None, transform_function=None,
                       transform_validation=False, skip_normalization=False):
    train_data, val_data, test_data, _ = load_mnist_datasets(
        val_ratio=val_ratio, noise_level=noise_level, transform_function=transform_function,
        transform_validation=transform_validation, skip_normalization=skip_normalization, seed=seed)

    if num_train_examples is not None:
        subset = np.random.choice(len(train_data), num_train_examples, replace=False)
        train_data = Subset(train_data, subset)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                            num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader


def load_cifar10_datasets(val_ratio=0.2, noise_level=0.0, skip_normalization=False, seed=42):
    data_dir = os.path.join(os.path.dirname(__file__), '../data/cifar10/')

    all_transforms = [transforms.ToTensor()]
    if not skip_normalization:
        # NOTE: this normalization is set so that models pretrained on ImageNet work well
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        all_transforms.append(normalize)
    composed_transform = transforms.Compose(all_transforms)

    train_data = datasets.CIFAR10(data_dir, download=True, train=True, transform=composed_transform)
    test_data = datasets.CIFAR10(data_dir, download=True, train=False, transform=composed_transform)
    train_data, val_data = split(train_data, val_ratio, seed)

    train_data.dataset_name = 'cifar10'
    val_data.dataset_name = 'cifar10'
    test_data.dataset_name = 'cifar10'

    # corrupt noise_level percent of the training labels
    is_corrupted = np.zeros(len(train_data), dtype=int)  # 0 clean, 1 corrupted, 2 accidentally correct
    for current_idx, sample_idx in enumerate(train_data.indices):
        if np.random.uniform(0, 1) < noise_level:
            new_label = np.random.randint(10)
            if new_label == train_data.dataset.targets[sample_idx]:
                is_corrupted[current_idx] = 2
            else:
                is_corrupted[current_idx] = 1
            train_data.dataset.targets[sample_idx] = new_label

    return train_data, val_data, test_data, is_corrupted


def load_cifar10_loaders(val_ratio=0.2, batch_size=128, noise_level=0.0, seed=42,
                         drop_last=False, num_train_examples=None, skip_normalization=False):
    train_data, val_data, test_data, _ = load_cifar10_datasets(val_ratio=val_ratio,
                                                               noise_level=noise_level,
                                                               skip_normalization=skip_normalization,
                                                               seed=seed)

    if num_train_examples is not None:
        subset = np.random.choice(len(train_data), num_train_examples, replace=False)
        train_data = Subset(train_data, subset)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=4, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                            num_workers=4, drop_last=drop_last)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader


def load_data_from_arguments(args, skip_normalization=False):
    """ Helper method for loading data from arguments.
    """
    transform_function = None
    if args.transform_function == 'remove_random_chunks':
        transform_function = create_remove_random_chunks_function(args.remove_prob)

    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = load_mnist_loaders(
            batch_size=args.batch_size, noise_level=args.noise_level,
            transform_function=transform_function,
            transform_validation=args.transform_validation,
            num_train_examples=args.num_train_examples,
            skip_normalization=skip_normalization)

    if args.dataset == 'cifar10':
        train_loader, val_loader, test_loader = load_cifar10_loaders(
            batch_size=args.batch_size, noise_level=args.noise_level,
            num_train_examples=args.num_train_examples,
            skip_normalization=skip_normalization)

    example_shape = train_loader.dataset[0][0].shape
    print("Dataset is loaded:\n\ttrain_samples: {}\n\tval_samples: {}\n\t"
          "test_samples: {}\n\tsample_shape: {}".format(
        len(train_loader.dataset), len(val_loader.dataset),
        len(test_loader.dataset), example_shape))

    return train_loader, val_loader, test_loader
