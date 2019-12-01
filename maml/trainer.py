import os
import json
from collections import defaultdict

import numpy as np
import torch

class Trainer(object):
    def __init__(self, meta_learner, meta_dataset, writer, log_interval,
                 save_interval, model_type, save_folder, total_iter):
        self._meta_learner = meta_learner
        self._meta_dataset = meta_dataset
        self._writer = writer
        self._log_interval = log_interval
        self._save_interval = save_interval
        self._model_type = model_type
        self._save_folder = save_folder
        self._total_iter = total_iter
        

    def run(self, is_training):
        if not is_training:
            all_pre_val_measurements = defaultdict(list)
            all_pre_train_measurements = defaultdict(list)
            all_post_val_measurements = defaultdict(list)
            all_post_train_measurements = defaultdict(list)

            # compute running accuracies for all datasets
            if self._meta_dataset.name == 'MultimodalFewShot':
                accuracies = [[] for i in range(self._meta_dataset.num_dataset)]

        for i, (train_tasks, val_tasks) in enumerate(
                iter(self._meta_dataset), start=1):

            # Save model
            if (i % self._save_interval == 0 or i == 1) and is_training:
                save_name = 'maml_{0}_{1}.pt'.format(self._model_type, i)
                save_path = os.path.join(self._save_folder, save_name)
                with open(save_path, 'wb') as f:
                    torch.save(self._meta_learner.state_dict(), f)

            (pre_train_measurements, adapted_params, embeddings
                ) = self._meta_learner.adapt(train_tasks)
            post_val_measurements = self._meta_learner.step(
                adapted_params, embeddings, val_tasks, is_training)

            # Tensorboard
            if (i % self._log_interval == 0 or i == 1):
                pre_val_measurements = self._meta_learner.measure(
                    tasks=val_tasks, embeddings_list=embeddings)
                post_train_measurements = self._meta_learner.measure(
                    tasks=train_tasks, adapted_params_list=adapted_params,
                    embeddings_list=embeddings)

                _grads_mean = np.mean(self._meta_learner._grads_mean)
                self._meta_learner._grads_mean = []

                self.log_output(
                    pre_val_measurements, pre_train_measurements,
                    post_val_measurements, post_train_measurements, 
                    i, _grads_mean)

                if is_training:
                    self.write_tensorboard(
                        pre_val_measurements, pre_train_measurements,
                        post_val_measurements, post_train_measurements, 
                        i, _grads_mean)

                if self._meta_dataset.name == 'MultimodalFewShot':

                    post_val_accuracies = self._meta_learner.measure_each(
                        tasks=val_tasks, adapted_params_list=adapted_params,
                        embeddings_list=embeddings)
                    
                    if is_training:
                        accuracies = [[] for i in range(self._meta_dataset.num_dataset)]
                    for i, accuracy in enumerate(post_val_accuracies):
                        accuracies[self._meta_dataset.dataset_names.index(
                            val_tasks[i].task_info)].append(accuracy)

                    accuracy_str = []
                    for i, accuracy in enumerate(accuracies):
                        accuracy_str.append('{}: {}'.format(
                            self._meta_dataset.dataset_names[i], 
                            'NaN' if len(accuracy) == 0 \
                                  else '{:.3f}%'.format(100*np.mean(accuracy))))

                    print('Individual accuracies: {}'.format('  '.join(accuracy_str)))
                    print('All accuracy: {:.3f}%'.format(100*np.mean(
                        [item for accuracy in accuracies for item in accuracy])))

            # Collect evaluation statistics over full dataset
            if not is_training:
                for key, value in sorted(pre_val_measurements.items()):
                    all_pre_val_measurements[key].append(value)
                for key, value in sorted(pre_train_measurements.items()):
                    all_pre_train_measurements[key].append(value)
                for key, value in sorted(post_val_measurements.items()):
                    all_post_val_measurements[key].append(value)
                for key, value in sorted(post_train_measurements.items()):
                    all_post_train_measurements[key].append(value)

        # Compute evaluation statistics assuming all batches were the same size
        if not is_training:
            results = {'num_batches': i}
            for key, value in sorted(all_pre_val_measurements.items()):
                results['pre_val_' + key] = value
            for key, value in sorted(all_pre_train_measurements.items()):
                results['pre_train_' + key] = value
            for key, value in sorted(all_post_val_measurements.items()):
                results['post_val_' + key] = value
            for key, value in sorted(all_post_train_measurements.items()):
                results['post_train_' + key] = value

            print('Evaluation results:')
            for key, value in sorted(results.items()):
                if not isinstance(value, int):
                    print('{}: {} +- {}'.format(
                        key, np.mean(value), self.compute_confidence_interval(value)))
                else:
                    print('{}: {}'.format(key, value))

            results_path = os.path.join(self._save_folder, 'results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f)

    def compute_confidence_interval(self, value):
        """
        Compute 95% +- confidence intervals over tasks
        change 1.960 to 2.576 for 99% +- confidence intervals
        """
        return np.std(value) * 1.960 / np.sqrt(len(value))

    def train(self):
        self.run(is_training=True)

    def eval(self):
        self.run(is_training=False)

    def write_tensorboard(self, pre_val_measurements, pre_train_measurements,
                          post_val_measurements, post_train_measurements, 
                          iteration, embedding_grads_mean=None):
        for key, value in pre_val_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_val'.format(key), value, iteration)
        for key, value in pre_train_measurements.items():
            self._writer.add_scalar(
                '{}/before_update/meta_train'.format(key), value, iteration)
        for key, value in post_train_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_train'.format(key), value, iteration)
        for key, value in post_val_measurements.items():
            self._writer.add_scalar(
                '{}/after_update/meta_val'.format(key), value, iteration)
        if embedding_grads_mean is not None:
            self._writer.add_scalar(
                'embedding_grads_mean', embedding_grads_mean, iteration)

    def log_output(self, pre_val_measurements, pre_train_measurements,
                   post_val_measurements, post_train_measurements, 
                   iteration, embedding_grads_mean=None):
        log_str = 'Iteration: {}/{} '.format(iteration, self._total_iter)
        for key, value in sorted(pre_val_measurements.items()):
            log_str = (log_str + '{} meta_val before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(pre_train_measurements.items()):
            log_str = (log_str + '{} meta_train before: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_train_measurements.items()):
            log_str = (log_str + '{} meta_train after: {:.3f} '
                                 ''.format(key, value))
        for key, value in sorted(post_val_measurements.items()):
            log_str = (log_str + '{} meta_val after: {:.3f} '
                                 ''.format(key, value))
        if embedding_grads_mean is not None:
            log_str = (log_str + 'embedding_grad_norm after: {:.3f} '
                                    ''.format(embedding_grads_mean))
        print(log_str)
