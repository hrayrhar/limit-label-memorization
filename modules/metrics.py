from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

from . import utils


class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError("name is not implemented")

    @abstractmethod
    def value(self, partition, **kwargs):
        raise NotImplementedError("value is not implemented")

    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        pass

    def on_iteration_end(self, *args, **kwargs):
        pass


class Accuracy(Metric):
    def __init__(self, output_key: str = 'pred', **kwargs):
        super(Accuracy, self).__init__(**kwargs)
        self.output_key = output_key

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(float)

    @property
    def name(self):
        return "accuracy"

    def value(self, partition, **kwargs):
        return self._accuracy[partition]

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        self._accuracy[partition] = accuracy
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, info, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(info[self.output_key]).argmax(axis=1).astype(np.int)
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)
        self._accuracy_storage[partition].append((pred == batch_labels).astype(np.float).mean())


class TopKAccuracy(Metric):
    def __init__(self, k, output_key: str = 'pred', **kwargs):
        super(TopKAccuracy, self).__init__(**kwargs)
        self.k = k
        self.output_key = output_key

        # initialize and use later
        self._accuracy_storage = defaultdict(list)
        self._accuracy = defaultdict(float)

    @property
    def name(self):
        return f"top{self.k}_accuracy"

    def value(self, partition, **kwargs):
        return self._accuracy[partition]

    def on_epoch_start(self, partition, **kwargs):
        self._accuracy_storage[partition] = []

    def on_epoch_end(self, partition, tensorboard, epoch, **kwargs):
        accuracy = np.mean(self._accuracy_storage[partition])
        tensorboard.add_scalar(f"metrics/{partition}_{self.name}", accuracy, epoch)

    def on_iteration_end(self, info, batch_labels, partition, **kwargs):
        pred = utils.to_numpy(info[self.output_key])
        batch_labels = utils.to_numpy(batch_labels[0]).astype(np.int)

        topk_predictions = np.argsort(-pred, axis=1)[:, :self.k]
        batch_labels = batch_labels.reshape((-1, 1)).repeat(self.k, axis=1)
        topk_correctness = (np.sum(topk_predictions == batch_labels, axis=1) >= 1)

        self._accuracy_storage[partition].append(topk_correctness.astype(np.float).mean())
