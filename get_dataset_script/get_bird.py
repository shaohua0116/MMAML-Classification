import subprocess
import os


cmds = []
cmds.append(['wget', 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'])
cmds.append(['tar', 'xvzf', 'CUB_200_2011.tgz'])
cmds.append(['python', 'get_dataset_script/proc_bird.py'])
cmds.append(['rm', '-rf', 'CUB_200_2011', 'CUB_200_2011.tgz'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)
