import subprocess


cmds = []
cmds.append(['wget', 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'])
cmds.append(['tar', '-xzvf', 'cifar-100-python.tar.gz'])
cmds.append(['mv', 'cifar-100-python', './data'])
cmds.append(['python3', 'get_dataset_script/proc_cifar.py'])
cmds.append(['rm', '-rf', 'cifar-100-python.tar.gz'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)
