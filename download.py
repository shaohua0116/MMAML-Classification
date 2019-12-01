import subprocess
import argparse


parser = argparse.ArgumentParser(description='Download datasets for MMAML.')
parser.add_argument('--dataset', metavar='N', type=str, nargs='+', 
                    choices=['aircraft', 'bird', 'cifar', 'miniimagenet'])


def download(dataset):
    cmd = ['python', 'get_dataset_script/get_{}.py'.format(dataset)]
    print(' '.join(cmd))
    subprocess.call(cmd)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    if len(args.dataset) > 0:
        for dataset in args.dataset:
            download(dataset)
