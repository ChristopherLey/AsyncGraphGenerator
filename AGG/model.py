from typing import Optional
from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from AGG.extended_typing import ContinuousTimeGraphSample


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
    ):
        if hidden_dim is None:
            hidden_dim = input_size
        if output_size is None:
            output_size = input_size
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)  # node normalisation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=embed_dim,
            vdim=embed_dim,
            batch_first=batch_first,
        )
        self.num_heads = num_heads
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(
            embed_dim, hidden_dim=4 * embed_dim, dropout=dropout
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ):
        h = self.norm1(x)
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
        attention, _ = self.self_attention(
            h, h, h, attn_mask=multihead_mask, key_padding_mask=key_padding_mask
        )
        x = x + attention
        x = x + self.feed_forward(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        target_dim: int,
        source_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        batch_first: bool = True,
    ):
        super().__init__()
        self.norm1_source = nn.LayerNorm(source_dim)  # node normalisation
        self.norm1_target = nn.LayerNorm(target_dim)  # node normalisation
        self.self_attention = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=source_dim,
            vdim=source_dim,
            batch_first=batch_first,
        )
        self.norm2 = nn.LayerNorm(target_dim)
        self.feed_forward = FeedForward(
            target_dim, hidden_dim=4 * target_dim, dropout=dropout
        )

    def forward(
        self, target: Tensor, source: Tensor, key_padding_mask: Optional[Tensor] = None
    ):
        key = self.norm1_source(source)
        mask = (torch.ones((target.shape[-2], source.shape[-2])) == 0).to(key.device)
        attention, _ = self.self_attention(
            query=self.norm1_target(target),
            key=key,
            value=key,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        x = target + attention
        x = x + self.feed_forward(self.norm2(x))
        return x


class AsynchronousGraphGenerator(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        num_heads,
        time_embedding_dim,
        num_node_types,
        type_embedding_dim,
        num_spatial_components,
        spatial_embedding_dim,
        num_categories,
        categorical_embedding_dim,
        dropout: float = 0.2,
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
        self.query_dim = time_embedding_dim + type_embedding_dim + spatial_embedding_dim
        self.time_embed = Time2Vec(time_embedding_dim)
        self.type_embed = nn.Embedding(num_node_types, type_embedding_dim)
        self.spatial_embed = nn.Embedding(num_spatial_components, spatial_embedding_dim)
        self.categorical_embedding = nn.Embedding(
            num_categories, categorical_embedding_dim
        )

        self.cross_attention = CrossAttentionBlock(
            self.query_dim, self.node_feature_dim, num_heads, dropout
        )
        self.graph_attention = SelfAttentionBlock(
            self.node_feature_dim, num_heads, dropout
        )
        self.cross_attention_2 = CrossAttentionBlock(
            self.query_dim, self.node_feature_dim, num_heads, dropout
        )
        self.head = FeedForward(self.query_dim, output_size=input_dim, dropout=dropout)

        self.mse = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample, device: torch.device = "cpu"
    ) -> Tuple[Tensor, Tensor]:

        features = self.feature_projection(graph.node_features.unsqueeze(-1).to(device))
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
        target = torch.cat(
            [
                self.time_embed(graph.target.time.unsqueeze(-1).to(device)),
                self.type_embed(graph.target.type_index.to(device)),
                self.spatial_embed(graph.target.spatial_index.to(device)),
            ],
            dim=-1,
        )
        key_padding_mask = graph.key_padding_mask.to(device)
        attn_mask = graph.attention_mask.to(device)
        h = self.graph_attention(source, attn_mask, key_padding_mask)
        y_hat = self.cross_attention(target, source, key_padding_mask)
        y_hat = self.cross_attention_2(y_hat, h, key_padding_mask)
        y_hat = self.head(y_hat)
        loss = self.mse(y_hat.squeeze(-1), graph.target.features.to(device))
        return loss, y_hat


if __name__ == "__main__":
    from Datasets.Beijing.datareader import AirQualityData
    from pathlib import Path
    from torch.utils.data import DataLoader
    from AGG.extended_typing import collate_graph_samples

    reader = AirQualityData(10, Path("../Datasets/Beijing/data/mongo_config.yaml"))
    print(len(reader))
    loader = DataLoader(reader, batch_size=2, collate_fn=collate_graph_samples)
    graph_sample = next(iter(loader))
    agg = AsynchronousGraphGenerator(
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
        dropout=0.0,
    )
    print(agg(graph_sample))
