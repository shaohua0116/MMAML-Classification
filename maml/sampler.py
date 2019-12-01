import random
from collections import defaultdict, namedtuple

from torch.utils.data.sampler import Sampler


class ClassBalancedSampler(Sampler):
    """Generates indices for class balanced batch by sampling with replacement.
    """
    def __init__(self, dataset_labels, num_classes_per_batch,
                 num_samples_per_class, num_total_batches, train):
        """
        Args:
            dataset_labels: list of dataset labels
            num_classes_per_batch: number of classes to sample for each batch
            num_samples_per_class: number of samples to sample for each class
                for each batch. For K shot learning this should be K + number
                of validation samples
            num_total_batches: total number of batches to generate
        """
        self._dataset_labels = dataset_labels
        self._classes = set(self._dataset_labels)
        self._class_to_samples = defaultdict(set)
        for i, c in enumerate(self._dataset_labels):
            self._class_to_samples[c].add(i)

        self._num_classes_per_batch = num_classes_per_batch
        self._num_samples_per_class = num_samples_per_class
        self._num_total_batches = num_total_batches
        self._train = train

    def __iter__(self):
        for i in range(self._num_total_batches):
            if len(self._class_to_samples.keys()) >= self._num_classes_per_batch:
                batch_classes = random.sample(
                    self._class_to_samples.keys(), self._num_classes_per_batch)
            else:
                batch_classes = [random.choice(list(self._class_to_samples.keys())) 
                                 for _ in range(self._num_classes_per_batch)]
            batch_samples = []
            for c in batch_classes:
                if len(self._class_to_samples[c]) >= self._num_samples_per_class:
                    class_samples = random.sample(
                        self._class_to_samples[c], self._num_samples_per_class)
                else:
                    class_samples = [random.choice(list(self._class_to_samples[c])) 
                                     for _ in range(self._num_samples_per_class)]

                for sample in class_samples:
                    batch_samples.append(sample)
            random.shuffle(batch_samples)
            for sample in batch_samples:
                yield sample

    def __len__(self):
        return self._num_total_batches
