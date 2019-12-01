import subprocess


cmds = []
cmds.append(['python', 'get_dataset_script/download_miniimagenet.py'])
cmds.append(['unzip', 'mini-imagenet.zip'])
cmds.append(['rm', '-rf', 'mini-imagenet.zip'])
cmds.append(['mkdir', 'miniimagenet'])
cmds.append(['mv', 'images', 'miniimagenet'])
cmds.append(['python', 'get_dataset_script/proc_miniimagenet.py'])
cmds.append(['mv', 'miniimagenet', './data'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)
