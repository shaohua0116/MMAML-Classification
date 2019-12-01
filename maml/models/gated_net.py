from collections import OrderedDict

import torch
import torch.nn.functional as F

from maml.models.model import Model
from IPython import embed

def weight_init(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0, std=0.01)
        module.bias.data.zero_()


class GatedNet(Model):
    def __init__(self, input_size, output_size, hidden_sizes=[40, 40],
                 nonlinearity=F.relu, condition_type='sigmoid_gate', condition_order='low2high'):
        super(GatedNet, self).__init__()
        self._nonlinearity = nonlinearity
        self._condition_type = condition_type
        self._condition_order = condition_order

        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, self.num_layers):
            self.add_module(
                'layer{0}_linear'.format(i),
                torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        self.add_module(
            'output_linear',
            torch.nn.Linear(layer_sizes[self.num_layers - 1],
                            layer_sizes[self.num_layers]))
        self.apply(weight_init)

    def conditional_layer(self, x, embedding):
        if self._condition_type == 'sigmoid_gate':
            x = x * F.sigmoid(embedding).expand_as(x)
        elif self._condition_type == 'affine':
            gammas, betas = torch.split(embedding, x.size(1), dim=-1)
            gammas = gammas + torch.ones_like(gammas)
            x = x * gammas + betas
        elif self._condition_type == 'softmax':
            x = x * F.softmax(embedding).expand_as(x)
        else:
            raise ValueError('Unrecognized conditional layer type {}'.format(
                self._condition_type))
        return x

    def forward(self, task, params=None, embeddings=None, training=True):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if embeddings is not None:
          if self._condition_order == 'high2low': ## High2Low
            embeddings = {'layer{}_linear'.format(len(params)-i): embedding
                            for i, embedding in enumerate(embeddings[::-1])}
          elif self._condition_order == 'low2high': ## Low2High
            embeddings = {'layer{}_linear'.format(i): embedding
                            for i, embedding in enumerate(embeddings[::-1], start=1)}
          else:
            raise NotImplementedError('Unsuppported order for using conditional layers')
        x = task.x.view(task.x.size(0), -1)

        for key, module in self.named_modules():
            if 'linear' in key:
                x = F.linear(x, weight=params[key + '.weight'],
                             bias=params[key + '.bias'])
                if 'output' not in key and embeddings is not None: # conditioning and nonlinearity
                    if type(embeddings.get(key, -1)) != type(-1):
                      x = self.conditional_layer(x, embeddings[key])

                    x = self._nonlinearity(x)

        return x
