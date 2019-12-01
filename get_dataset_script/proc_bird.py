import numpy as np
import os
import subprocess
from imageio import imread, imwrite


source_dir = './CUB_200_2011/images'
target_dir = './data/bird'

percentage_train_class = 70
percentage_val_class = 15
percentage_test_class = 15
train_val_test_ratio = [
    percentage_train_class, percentage_val_class, percentage_test_class]

classes = os.listdir(source_dir)

rs = np.random.RandomState(123)
rs.shuffle(classes)
num_train, num_val, num_test = [
    int(float(ratio)/np.sum(train_val_test_ratio)*len(classes))
    for ratio in train_val_test_ratio]

classes = {
    'train': classes[:num_train],
    'val': classes[num_train:num_train+num_val],
    'test': classes[num_train+num_val:]
}

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
for k in classes.keys():
    target_dir_k = os.path.join(target_dir, k)
    if not os.path.exists(target_dir_k):
        os.makedirs(target_dir_k)
    cmd = ['mv'] + [os.path.join(source_dir, c) for c in classes[k]] + [target_dir_k]
    subprocess.call(cmd)

_ids = []

for root, dirnames, filenames in os.walk(target_dir):
    for filename in filenames:
        if filename.endswith(('.jpg', '.webp', '.JPEG', '.png', 'jpeg')):
            _ids.append(os.path.join(root, filename))

for path in _ids:
    try:
        img = imread(path)
    except:
        print(img)
    if len(img.shape) < 3:
        print(path)
        img = np.tile(np.expand_dims(img, axis=-1), [1, 1, 3])
        imwrite(path, img)
    else:
        if img.shape[-1] == 1:
            print(path)
            img = np.tile(img, [1, 1, 3])
            imwrite(path, img)

# resize images
cmd = ['python', 'get_dataset_script/resize_dataset.py', './data/bird']
subprocess.call(cmd)
