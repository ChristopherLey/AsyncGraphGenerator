from typing import Optional

import torch
from torch import nn
from torch import Tensor


class Time2Vec(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, tau: Tensor) -> Tensor:
        time_scaling = self.linear(tau)
        if len(tau.shape) == 2:
            periodic_embedding = torch.sin(time_scaling[:, 1:])
            time_embedding = torch.cat(
                (time_scaling[:, 0:1], periodic_embedding), dim=1
            )
        else:
            periodic_embedding = torch.sin(time_scaling[:, :, 1:])
            time_embedding = torch.cat(
                (time_scaling[:, :, 0:1], periodic_embedding), dim=2
            )
        return time_embedding


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: Optional[int] = None,
        output_size: Optional[int] = None,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
    ):
        if hidden_dim is None:
            hidden_dim = input_size
        if output_size is None:
            output_size = input_size
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        return self.ff(x)
