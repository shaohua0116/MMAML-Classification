import numpy as np
import os
import subprocess


source_dir = './fgvc-aircraft-2013b/data'
target_dir = './data/aircraft'

percentage_train_class = 70
percentage_val_class = 15
percentage_test_class = 15
train_val_test_ratio = [
    percentage_train_class, percentage_val_class, percentage_test_class]

with open(os.path.join(source_dir, 'variants.txt')) as f:
    lines = f.readlines()
    classes = [line.strip() for line in lines]

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
    for c in classes[k]:
        c = c.replace('/', '-')
        target_dir_k_c = os.path.join(target_dir_k, c)
        if not os.path.exists(target_dir_k_c):
            os.makedirs(target_dir_k_c)

lines = []
with open(os.path.join(source_dir, 'images_variant_trainval.txt')) as f:
    lines += f.readlines()
with open(os.path.join(source_dir, 'images_variant_test.txt')) as f:
    lines += f.readlines()
lines = [line.strip() for line in lines]

for i, line in enumerate(lines):
    image_num, image_class = line.split(' ', 1)
    image_class = image_class.replace('/', '-')
    image_k = list(classes.keys())[np.argmax([image_class in classes[k] for k in list(classes.keys())])]
    image_source_path = os.path.join(source_dir, 'images', '{}.jpg'.format(image_num))
    image_target_path = os.path.join(target_dir, image_k, image_class)
    cmd = ['mv', image_source_path, image_target_path]
    subprocess.call(cmd)
    print('{}/{} {}'.format(i, len(lines), ' '.join(cmd)))

# resize images
cmd = ['python', 'get_dataset_script/resize_dataset.py', './data/aircraft']
subprocess.call(cmd)
