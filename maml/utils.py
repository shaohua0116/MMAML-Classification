import subprocess

import torch


def accuracy(preds, y):
    _, preds = torch.max(preds.data, 1)
    total = y.size(0)
    correct = (preds == y).sum().float()
    return correct / total


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_git_revision_hash():
    return str(subprocess.check_output(['git', 'rev-parse', 'HEAD']))