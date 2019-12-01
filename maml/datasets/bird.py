import glob
import random
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from maml.sampler import ClassBalancedSampler
from maml.datasets.metadataset import Task


class BirdMAMLSplit():
    def __init__(self, root, train=True, num_train_classes=140,
                 transform=None, target_transform=None, **kwargs):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root + '/bird'

        self._train = train

        if self._train:
            all_character_dirs = glob.glob(self.root + '/train/**')
            self._characters = all_character_dirs
        else:
            all_character_dirs = glob.glob(self.root + '/test/**')
            self._characters = all_character_dirs

        self._character_images = []
        for i, char_path in enumerate(self._characters):
            img_list = [(cp, i) for cp in glob.glob(char_path + '/*')]
            self._character_images.append(img_list)

        self._flat_character_images = sum(self._character_images, [])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target
            character class.
        """
        image_path, character_class = self._flat_character_images[index]
        image = Image.open(image_path, mode='r')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class


class BirdMetaDataset(object):
    """
    TODO: Check if the data loader is fast enough.
    Args:
        root: path to bird dataset
        img_side_len: images are scaled to this size
        num_classes_per_batch: number of classes to sample for each batch
        num_samples_per_class: number of samples to sample for each class
            for each batch. For K shot learning this should be K + number
            of validation samples
        num_total_batches: total number of tasks to generate
        train: whether to create data loader from the test or validation data
    """
    def __init__(self, name='Bird', root='data',
                 img_side_len=84, img_channel=3,
                 num_classes_per_batch=5, num_samples_per_class=6,
                 num_total_batches=200000,
                 num_val_samples=1, meta_batch_size=40, train=True,
                 num_train_classes=1100, num_workers=0, device='cpu'):
        self.name = name
        self._root = root
        self._img_side_len = img_side_len
        self._img_channel = img_channel
        self._num_classes_per_batch = num_classes_per_batch
        self._num_samples_per_class = num_samples_per_class
        self._num_total_batches = num_total_batches
        self._num_val_samples = num_val_samples
        self._meta_batch_size = meta_batch_size
        self._num_train_classes = num_train_classes
        self._train = train
        self._num_workers = num_workers
        self._device = device

        self._total_samples_per_class = (
            num_samples_per_class + num_val_samples)
        self._dataloader = self._get_bird_data_loader()

        self.input_size = (img_channel, img_side_len, img_side_len)
        self.output_size = self._num_classes_per_batch

    def _get_bird_data_loader(self):
        assert self._img_channel == 1 or self._img_channel == 3
        resize = transforms.Resize(
            (self._img_side_len, self._img_side_len), Image.LANCZOS)
        if self._img_channel == 1:
            img_transform = transforms.Compose(
                [resize, transforms.Grayscale(num_output_channels=1),
                 transforms.ToTensor()])
        else:
            img_transform = transforms.Compose(
                [resize, transforms.ToTensor()])
        dset = BirdMAMLSplit(
            self._root, transform=img_transform, train=self._train,
            download=True, num_train_classes=self._num_train_classes)
        _, labels = zip(*dset._flat_character_images)
        sampler = ClassBalancedSampler(labels, self._num_classes_per_batch,
                                       self._total_samples_per_class,
                                       self._num_total_batches, self._train)

        batch_size = (self._num_classes_per_batch *
                      self._total_samples_per_class *
                      self._meta_batch_size)
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler,
                            num_workers=self._num_workers, pin_memory=True)
        return loader

    def _make_single_batch(self, imgs, labels):
        """Split imgs and labels into train and validation set.
        TODO: check if this might become the bottleneck"""
        # relabel classes randomly
        new_labels = list(range(self._num_classes_per_batch))
        random.shuffle(new_labels)
        labels = labels.tolist()
        label_set = set(labels)
        label_map = {label: new_labels[i] for i, label in enumerate(label_set)}
        labels = [label_map[l] for l in labels]

        label_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_indices[label].append(i)

        # assign samples to train and validation sets
        val_indices = []
        train_indices = []
        for label, indices in label_indices.items():
            val_indices.extend(indices[:self._num_val_samples])
            train_indices.extend(indices[self._num_val_samples:])
        label_tensor = torch.tensor(labels, device=self._device)
        imgs = imgs.to(self._device)
        train_task = Task(imgs[train_indices], label_tensor[train_indices], self.name)
        val_task = Task(imgs[val_indices], label_tensor[val_indices], self.name)

        return train_task, val_task

    def _make_meta_batch(self, imgs, labels):
        batches = []
        inner_batch_size = (
            self._total_samples_per_class * self._num_classes_per_batch)
        for i in range(0, len(imgs) - 1, inner_batch_size):
            batch_imgs = imgs[i:i+inner_batch_size]
            batch_labels = labels[i:i+inner_batch_size]
            batch = self._make_single_batch(batch_imgs, batch_labels)
            batches.append(batch)

        train_tasks, val_tasks = zip(*batches)

        return train_tasks, val_tasks

    def __iter__(self):
        for imgs, labels in iter(self._dataloader):
            train_tasks, val_tasks = self._make_meta_batch(imgs, labels)
            yield train_tasks, val_tasks
