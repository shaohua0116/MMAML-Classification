import torch

class GRUEmbeddingModel(torch.nn.Module):
    def __init__(self, input_size, output_size, embedding_dims,
                 hidden_size=40, num_layers=2):
        super(GRUEmbeddingModel, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._embedding_dims = embedding_dims
        self._bidirectional = True
        self._device = 'cpu'

        rnn_input_size = int(input_size + output_size)
        self.rnn = torch.nn.GRU(rnn_input_size, hidden_size, num_layers, bidirectional=self._bidirectional)

        self._embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self._embeddings.append(torch.nn.Linear(hidden_size*(2 if self._bidirectional else 1), dim))

    def forward(self, task):
        batch_size = 1
        h0 = torch.zeros(self._num_layers*(2 if self._bidirectional else 1),
                         batch_size, self._hidden_size, device=self._device)

        x = task.x.view(task.x.size(0), -1)
        y = task.y.view(task.y.size(0), -1)

        # LSTM input dimensions are seq_len, batch, input_size
        inputs = torch.cat((x, y), dim=1).view(x.size(0), 1, -1)
        output, _ = self.rnn(inputs, h0)
        if self._bidirectional:
          N, B, H = output.shape
          output = output.view(N, B, 2, H // 2)
          embedding_input = torch.cat([output[-1, :, 0], output[0, :, 1]], dim=1)

        out_embeddings = []
        for embedding in self._embeddings:
            out_embeddings.append(embedding(embedding_input))
        return out_embeddings

    def to(self, device, **kwargs):
        self._device = device
        super(GRUEmbeddingModel, self).to(device, **kwargs)
