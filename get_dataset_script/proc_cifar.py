import torch
import numpy as np
import pickle
import os

img_size=32
classes={
    'train': [1, 2,  3,  4, 5, 6, 9, 10, 15, 17, 18, 19],
    'val':   [8, 11, 13, 16],
    'test':  [0, 7,  12, 14]
  }

def _get_file_path(filename=""):
  return os.path.join('./data', "cifar-100-python/", filename)

def _unpickle(filename):
  """
  Unpickle the given file and return the data.
  Note that the appropriate dir-name is prepended the filename.
  """

  # Create full path for the file.
  file_path = _get_file_path(filename)

  print("Loading data: " + file_path)

  with open(file_path, mode='rb') as file:
    # In Python 3.X it is important to set the encoding,
    # otherwise an exception is raised here.
    data = pickle.load(file, encoding='latin1')

  return data

# import IPython
# IPython.embed()
meta  = _unpickle('meta')
train = _unpickle('train')
test  = _unpickle('test')

data = np.concatenate([train['data'], test['data']])
labels = np.array(train['fine_labels'] + test['fine_labels'])
filts = np.array(train['coarse_labels'] + test['coarse_labels'])

cifar_data = {}
cifar_label = {}
for k, v in classes.items():
  data_filter = np.zeros_like(filts)
  for i in v: data_filter += ( filts == i )
  assert data_filter.max() == 1

  cifar_data[k] = data[data_filter == 1]
  cifar_label[k] = labels[data_filter == 1]

torch.save({'data': cifar_data, 'label': cifar_label}, './data/cifar100.pth')
