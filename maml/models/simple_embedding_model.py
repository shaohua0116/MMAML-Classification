import torch

class SimpleEmbeddingModel(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dims):
        super(SimpleEmbeddingModel, self).__init__()
        self._embeddings = torch.nn.ModuleList()
        for dim in embedding_dims:
            self._embeddings.append(torch.nn.Embedding(num_embeddings, dim))
        self._device = 'cpu'

    def forward(self, task):
        task_id = torch.tensor(task.task_id, dtype=torch.long,
                               device=self._device)
        out_embeddings = []
        for embedding in self._embeddings:
            out_embeddings.append(embedding(task_id))
        return out_embeddings

    def to(self, device, **kwargs):
        self._device = device
        super(SimpleEmbeddingModel, self).to(device, **kwargs)