import subprocess


cmds = []
cmds.append(['wget', 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'])
cmds.append(['tar', 'xvzf', 'fgvc-aircraft-2013b.tar.gz'])
cmds.append(['python', 'get_dataset_script/proc_aircraft.py'])
cmds.append(['rm', '-rf', 'fgvc-aircraft-2013b.tar.gz', 'fgvc-aircraft-2013b'])

for cmd in cmds:
    print(' '.join(cmd))
    subprocess.call(cmd)
