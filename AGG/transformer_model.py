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
from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.utils import FeedForward
from AGG.utils import Time2Vec


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_drop: float = 0.2,
        dropout: float = 0.2,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # node normalisation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_drop,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=batch_first,
        )
        self.num_heads = num_heads
        self.feed_forward = FeedForward(
            input_size=embed_dim,
            hidden_dim=embed_dim * num_heads,
            output_size=embed_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if len(attention_mask.shape) == 3:
            B, N, _ = attention_mask.shape
            multihead_mask = (
                attention_mask.unsqueeze(0)
                .repeat(self.num_heads, 1, 1, 1)
                .transpose(1, 0)
                .reshape(-1, N, N)
            )
        else:
            multihead_mask = attention_mask
        attention, attention_weights = self.self_attention(
            x, x, x, attn_mask=multihead_mask, key_padding_mask=key_padding_mask
        )
        x = x + attention
        x = x + self.feed_forward(self.norm1(x))
        return self.norm2(x), attention_weights


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        target_dim: int,
        source_dim: int,
        num_heads: int,
        attention_drop: float = 0.2,
        dropout: float = 0.2,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm1_source = nn.LayerNorm(source_dim)  # node normalisation
        self.norm1_target = nn.LayerNorm(target_dim)  # node normalisation
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            dropout=attention_drop,
            kdim=source_dim,
            vdim=source_dim,
            batch_first=batch_first,
        )
        self.norm2 = nn.LayerNorm(target_dim)
        self.feed_forward = FeedForward(
            input_size=target_dim,
            hidden_dim=target_dim * num_heads,
            output_size=target_dim,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(target_dim)

    def forward(
        self, target: Tensor, source: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        key = self.norm1_source(source)
        mask = (torch.ones((target.shape[-2], source.shape[-2])) == 0).to(key.device)
        attention, attention_weights = self.cross_attention(
            query=self.norm1_target(target),
            key=key,
            value=key,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        x = target + attention
        x = x + self.feed_forward(self.norm2(x))
        return self.norm3(x), attention_weights


class AsynchronousGraphGeneratorTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        num_heads: int,
        time_embedding_dim: int,
        num_node_types: int,
        type_embedding_dim: int,
        num_spatial_components: int,
        spatial_embedding_dim: int,
        num_categories: int,
        categorical_embedding_dim: int,
        num_layers: int,
        attention_drop: float = 0.2,
        dropout: float = 0.2,
        query_includes_categorical: bool = False
    ):
        super().__init__()
        self.node_feature_dim = (
            feature_dim
            + time_embedding_dim
            + type_embedding_dim
            + spatial_embedding_dim
            + categorical_embedding_dim
        )
        self.feature_projection = nn.Linear(input_dim, feature_dim)
        if query_includes_categorical:
            self.query_includes_categorical = True
            self.query_dim = time_embedding_dim + type_embedding_dim + spatial_embedding_dim + categorical_embedding_dim
        else:
            self.query_includes_categorical = False
            self.query_dim = time_embedding_dim + type_embedding_dim + spatial_embedding_dim
        self.time_embed = Time2Vec(time_embedding_dim)
        self.type_embed = nn.Embedding(num_node_types, type_embedding_dim)
        self.spatial_embed = nn.Embedding(num_spatial_components, spatial_embedding_dim)
        self.categorical_embedding = nn.Embedding(
            num_categories, categorical_embedding_dim
        )
        self.agg_layers = nn.ModuleList()
        for i in range(num_layers):
            self.agg_layers.append(
                SelfAttentionBlock(
                    embed_dim=self.node_feature_dim,
                    num_heads=num_heads,
                    dropout=attention_drop,
                    batch_first=True,
                )
            )
        self.cross_attention = CrossAttentionBlock(
            target_dim=self.query_dim,
            source_dim=self.node_feature_dim,
            num_heads=num_heads,
            dropout=attention_drop,
        )
        self.head = FeedForward(
            input_size=self.query_dim,
            hidden_dim=self.query_dim * num_heads,
            output_size=input_dim,
            dropout=dropout,
        )

    def forward(
        self, graph: ContinuousTimeGraphSample, device: torch.device = "cpu"
    ) -> Tuple[Tensor, list]:
        if len(graph.node_features.shape) < 3:
            features = self.feature_projection(graph.node_features.unsqueeze(-1).to(device))
        elif len(graph.node_features.shape) == 3:
            features = self.feature_projection(graph.node_features.to(device))
        time_encode = self.time_embed(graph.time.unsqueeze(-1).to(device))
        type_encode = self.type_embed(graph.type_index.to(device))
        spatial_encode = self.spatial_embed(graph.spatial_index.to(device))
        source_list = [features, time_encode, type_encode, spatial_encode]
        if graph.category_index is not None:
            categorical_encode = self.categorical_embedding(
                graph.category_index.to(device)
            )
            source_list.append(categorical_encode)
        source = torch.cat(
            source_list,
            dim=-1,
        )
        query_list = [
                self.time_embed(graph.target.time.unsqueeze(-1).to(device)),
                self.type_embed(graph.target.type_index.to(device)),
                self.spatial_embed(graph.target.spatial_index.to(device)),
            ]
        if self.query_includes_categorical:
            query_list.append(
                self.categorical_embedding(
                    graph.target.category_index.unsqueeze(-1).to(device)
                )
            )
        target = torch.cat(
            query_list,
            dim=-1,
        )
        key_padding_mask = graph.key_padding_mask.to(device)
        attn_mask = graph.attention_mask.to(device)
        hidden = source
        total_attention = []
        for agg_layer in self.agg_layers:
            hidden, attention_weights = agg_layer(hidden, attn_mask, key_padding_mask)
            total_attention.append(attention_weights)
        y_hat, attention_weights = self.cross_attention(
            target, hidden, key_padding_mask
        )
        total_attention.append(attention_weights)
        y_hat = self.head(y_hat)
        y_hat = y_hat.squeeze(-1)
        return y_hat, total_attention


if __name__ == "__main__":
    from Datasets.Beijing.datareader import AirQualityData
    from pathlib import Path
    from torch.utils.data import DataLoader
    from AGG.extended_typing import collate_graph_samples

    reader = AirQualityData(10, Path("../Datasets/Beijing/data/mongo_config.yaml"))
    print(len(reader))
    loader = DataLoader(reader, batch_size=2, collate_fn=collate_graph_samples)
    graph_sample = next(iter(loader))
    agg = AsynchronousGraphGeneratorTransformer(
        input_dim=1,
        feature_dim=10,
        num_heads=5,
        time_embedding_dim=10,
        num_node_types=len(reader.type_index),
        type_embedding_dim=10,
        num_spatial_components=len(reader.spatial_index),
        spatial_embedding_dim=10,
        num_categories=len(reader.category_index),
        categorical_embedding_dim=10,
        num_layers=3,
        dropout=0.0,
    )
    print(agg(graph_sample))
