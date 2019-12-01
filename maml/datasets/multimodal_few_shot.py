import numpy as np
from itertools import chain
from maml.datasets.metadataset import Task


class MultimodalFewShotDataset(object):

    def __init__(self, datasets, num_total_batches,
                 name='MultimodalFewShot',
                 mix_meta_batch=True, mix_mini_batch=False,
                 train=True, verbose=False, txt_file=None):
        self._datasets = datasets
        self._num_total_batches = num_total_batches
        self.name = name
        self.num_dataset = len(datasets)
        self.dataset_names = [dataset.name for dataset in self._datasets]
        self._meta_batch_size = datasets[0]._meta_batch_size
        self._mix_meta_batch = mix_meta_batch
        self._mix_mini_batch = mix_mini_batch
        self._train = train
        self._verbose = verbose
        self._txt_file = open(txt_file, 'w') if not txt_file is None else None
        
        # make sure all input/output sizes match
        input_size_list = [dataset.input_size for dataset in self._datasets]
        assert input_size_list.count(input_size_list[0]) == len(input_size_list)
        output_size_list = [dataset.output_size for dataset in self._datasets]
        assert output_size_list.count(output_size_list[0]) == len(output_size_list)
        self.input_size = datasets[0].input_size
        self.output_size = datasets[0].output_size

        # build iterators
        self._datasets_iter = [iter(dataset) for dataset in self._datasets]
        self._iter_index = 0

        # print info
        print('Multimodal Few Shot Datasets: {}'.format(' '.join(self.dataset_names)))
        print('mix meta batch: {}'.format(mix_meta_batch))
        print('mix mini batch: {}'.format(mix_mini_batch))

    def __next__(self):
        if self.n < self._num_total_batches: 
            if not self._mix_meta_batch and not self._mix_mini_batch:
                dataset_index = np.random.randint(len(self._datasets))
                if self._verbose:
                    print('Sample from: {}'.format(self._datasets[dataset_index].name))
                train_tasks, val_tasks = next(self._datasets_iter[dataset_index])
                return train_tasks, val_tasks
            else: 
                # get all tasks
                tasks = []
                all_train_tasks = []
                all_val_tasks = []
                for dataset_iter in self._datasets_iter:
                    train_tasks, val_tasks = next(dataset_iter)
                    all_train_tasks.extend(train_tasks)
                    all_val_tasks.extend(val_tasks)
                
                if not self._mix_mini_batch:
                    # mix them to obtain a meta batch
                    """
                    # randomly sample task
                    dataset_indexes = np.random.choice(
                        len(all_train_tasks), size=self._meta_batch_size, replace=False)
                    """
                    # balancedly sample from all datasets
                    dataset_indexes = []
                    if self._train:
                        dataset_start_idx = np.random.randint(0, self.num_dataset)
                    else:
                        dataset_start_idx = (self._iter_index + self._meta_batch_size) % self.num_dataset
                        self._iter_index += self._meta_batch_size
                        self._iter_index = self._iter_index % self.num_dataset

                    for i in range(self._meta_batch_size):
                        dataset_indexes.append(
                            np.random.randint(0, self._meta_batch_size)+
                            ((i+dataset_start_idx)%self.num_dataset)*self._meta_batch_size)

                    train_tasks = []
                    val_tasks = []
                    dataset_names = []
                    for dataset_index in dataset_indexes:
                        train_tasks.append(all_train_tasks[dataset_index])
                        val_tasks.append(all_val_tasks[dataset_index])
                        dataset_names.append(self._datasets[dataset_index//self._meta_batch_size].name)
                    if self._verbose:
                        print('Sample from: {} (indexes: {})'.format(
                            [name for name in dataset_names], dataset_indexes))
                    if self._txt_file is not None:
                        for name in dataset_names:
                            self._txt_file.write(name+'\n')
                    return train_tasks, val_tasks
                else:
                    # mix them to obtain a mini batch and make a meta batch
                    raise NotImplementedError
        else:
            raise StopIteration

    def __iter__(self):
        self.n = 0
        return self
