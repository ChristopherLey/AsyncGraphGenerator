"""
    Copyright (C) 2023, Christopher Paul Ley
    Asynchronous Graph Generator

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import io
from typing import Optional

import torch
from PIL import Image
from torch import nn
from torch import Tensor
from torchvision import transforms


class FrequencyScale(nn.Module):
    def __init__(self, embedding_dim: int, include_linear: bool = True, frequency_scaling: float = 2.0 * torch.pi):
        super().__init__()
        self.frequency_scaling = frequency_scaling
        self.linear = nn.Linear(1, embedding_dim, bias=False)
        self.bias = nn.parameter.Parameter(nn.init.uniform_(torch.empty(embedding_dim), -1.0, 1.0))
        self.register_parameter("bias", self.bias)
        self.include_linear = include_linear
        self.weight = self.linear.weight

    def forward(self, tau: Tensor) -> Tensor:
        time_scaling = self.linear(tau)
        if not self.include_linear:
            time_scaled = self.frequency_scaling * time_scaling + self.bias
        else:
            if len(tau.shape) == 2:
                pi_scaled = self.frequency_scaling * time_scaling[:, 1:] + self.bias[1:]
                time_scaled = torch.cat(
                    (time_scaling[:, 0:1] + self.bias[0:1], pi_scaled), dim=1
                )
            else:
                pi_scaled = self.frequency_scaling * time_scaling[:, :, 1:] + self.bias[1:]
                time_scaled = torch.cat(
                    (time_scaling[:, :, 0:1] + self.bias[0:1], pi_scaled), dim=2
                )
        return time_scaled

    def extra_repr(self) -> str:
        return f"(unscaled_bias): Parameter(in_features={self.bias.shape[0]})"


class Time2Vec(nn.Module):
    """
    Time2Vec layer from the paper "Time2Vec: Learning a Vector Representation of Time"
    """
    def __init__(self, embedding_dim: int, include_linear: bool = True, frequency_scaling: float = torch.pi*2.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = FrequencyScale(embedding_dim, include_linear, frequency_scaling)
        self.include_linear = include_linear
        self.frequency_scaling = frequency_scaling

    def forward(self, tau: Tensor) -> Tensor:
        if len(tau.shape) == 1:
            tau = tau.unsqueeze(-1)
        time_scaling = self.scale(tau)
        if len(tau.shape) == 2:
            if not self.include_linear:
                time_embedding = torch.sin(time_scaling)
            else:
                periodic_embedding = torch.sin(time_scaling[:, 1:])
                time_embedding = torch.cat(
                    (time_scaling[:, 0:1], periodic_embedding), dim=1
                )
        else:
            if not self.include_linear:
                time_embedding = torch.sin(time_scaling)
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


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    T = transforms.ToTensor()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return T(img)


if __name__ == "__main__":
    t2v = Time2Vec(4)
    print(t2v)
