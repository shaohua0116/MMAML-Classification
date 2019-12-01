import os
import glob
import random
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.utils import list_files

from maml.sampler import ClassBalancedSampler
from maml.datasets.metadataset import Task

class Cifar100MAMLSplit():
    def __init__(self, root, train=True, num_train_classes=100,
                 transform=None, target_transform=None, **kwargs):
        self.transform = transform
        self.target_transform = target_transform
        self.root = os.path.join(root, 'cifar100.pth')

        self._train = train
        self._num_train_classes = num_train_classes

        self._dataset = torch.load(self.root)
        if self._train:
            self._images = torch.FloatTensor(self._dataset['data']['train'].reshape([-1, 3, 32, 32]))
            self._labels = torch.LongTensor(self._dataset['label']['train'])
        else:
            self._images = torch.FloatTensor(self._dataset['data']['test'].reshape([-1, 3, 32, 32]))
            self._labels = torch.LongTensor(self._dataset['label']['test'])

    def __getitem__(self, index):
        image = self._images[index]

        if self.transform:
            image = self.transform(self._images[index])

        return image, self._labels[index]

class Cifar100MetaDataset(object):
    def __init__(self, name='FC100', root='data', 
                 img_side_len=32, img_channel=3,
                 num_classes_per_batch=5, num_samples_per_class=6, 
                 num_total_batches=200000,
                 num_val_samples=1, meta_batch_size=32, train=True,
                 num_train_classes=100, num_workers=0, device='cpu'):
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

        self._total_samples_per_class = (num_samples_per_class + num_val_samples)
        self._dataloader = self._get_cifar100_data_loader()

        self.input_size = (img_channel, img_side_len, img_side_len)
        self.output_size = self._num_classes_per_batch

    def _get_cifar100_data_loader(self):
        assert self._img_channel == 1 or self._img_channel == 3
        to_imgae = transforms.ToPILImage()
        resize = transforms.Resize(self._img_side_len, Image.LANCZOS)
        if self._img_channel == 1:
            img_transform = transforms.Compose(
                [to_imgae, resize, 
                 transforms.Grayscale(num_output_channels=1),
                 transforms.ToTensor()])
        else:
            img_transform = transforms.Compose(
                [to_imgae, resize, transforms.ToTensor()])
        dset = Cifar100MAMLSplit(self._root, transform=img_transform,
                                 train=self._train, download=True,
                                 num_train_classes=self._num_train_classes)
        labels = dset._labels.numpy().tolist()
        sampler = ClassBalancedSampler(labels, self._num_classes_per_batch,
                                       self._total_samples_per_class,
                                       self._num_total_batches, self._train)

        batch_size = (self._num_classes_per_batch
                      * self._total_samples_per_class
                      * self._meta_batch_size)
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
        inner_batch_size = (self._total_samples_per_class
                            * self._num_classes_per_batch)
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
