import torch
from torch import nn


class Time2Vec(nn.Module):
    def __init__(self, feature_length: int):
        super(Time2Vec, self).__init__()
        assert feature_length >= 1, "a feature length >= 1 is required"
        self.phase = nn.Linear(1, out_features=feature_length)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t2v = self.phase(t)
        if len(t2v.shape) == 1:
            t2v[1:] = torch.sin(t2v[1:])
        else:  # batched
            t2v[:, 1:] = torch.sin(t2v[:, 1:])
        return t2v


class TemporalGraphNetwork(nn.Module):
    def __init__(
        self, time_encode_dim: int = 3, sensor_embed_dim: int = 8, num_heads: int = 6
    ):
        super(TemporalGraphNetwork, self).__init__()
        self.sensor_embedding = nn.Embedding(6, sensor_embed_dim)
        self.time_encoding = Time2Vec(time_encode_dim)
        self.num_heads = num_heads
        self.node_size = sensor_embed_dim * time_encode_dim
        node_embed_dim = self.node_size * num_heads
        self.attention = nn.MultiheadAttention(node_embed_dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(self.node_size * (num_heads + 1), self.node_size),
            nn.ELU(),
            nn.Linear(self.node_size, 1),
        )

    def message(self, x):
        pass

    def aggregation(self, x):
        pass

    def memory_update(self):
        pass

    def encoding(self):
        pass

    def forward(self, x):
        pass
