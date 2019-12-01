from collections import OrderedDict

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    @property
    def param_dict(self):
        return OrderedDict(self.named_parameters())