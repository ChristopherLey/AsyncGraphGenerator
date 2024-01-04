"""
# Graph Attention Networks v2 (GATv2)
This is a [PyTorch](https://pytorch.org) implementation of the GATv2 operator from the paper
[How Attentive are Graph Attention Networks?](https://arxiv.org/abs/2105.14491).

GATv2s work on graph data similar to [GAT](../gat/index.html).
A graph consists of nodes and edges connecting nodes.
For example, in Cora dataset the nodes are research papers and the edges are citations that
connect the papers.

The GATv2 operator fixes the static attention problem of the standard [GAT](../gat/index.html).
Static attention is when the attention to the key nodes has the same rank (order) for any query node.
[GAT](../gat/index.html) computes attention from query node $i$ to key node $j$ as,

\begin{align}
e_{ij} &= \text{LeakyReLU} \Big(\mathbf{a}^\top \Big[
 \mathbf{W} \overrightarrow{h_i} \Vert  \mathbf{W} \overrightarrow{h_j}
\Big] \Big) \\
&=
\text{LeakyReLU} \Big(\mathbf{a}_1^\top  \mathbf{W} \overrightarrow{h_i} +
 \mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}
\Big)
\end{align}

Note that for any query node $i$, the attention rank ($argsort$) of keys depends only
on $\mathbf{a}_2^\top  \mathbf{W} \overrightarrow{h_j}$.

Therefore, the attention rank of keys remains the same (*static*) for all queries.

GATv2 allows dynamic attention by changing the attention mechanism,

\begin{align}
e_{ij} &= \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W} \Big[
 \overrightarrow{h_i} \Vert  \overrightarrow{h_j}
\Big] \Big) \\
&= \mathbf{a}^\top \text{LeakyReLU} \Big(
\mathbf{W}_l \overrightarrow{h_i} +  \mathbf{W}_r \overrightarrow{h_j}
\Big)
\end{align}

The paper shows that GATs static attention mechanism fails on some graph problems
with a synthetic dictionary lookup dataset.
It's a fully connected bipartite graph where one set of nodes (query nodes)
have a key associated with it
and the other set of nodes have both a key and a value associated with it.
The goal is to predict the values of query nodes.
GAT fails on this task because of its limited static attention.
"""
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.utils import FeedForward
from AGG.utils import Time2Vec


class GraphAttentionLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False,
                 is_concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.share_weights = share_weights
        self.is_concat = is_concat

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
        self.linear_source = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        if share_weights:
            self.linear_target = self.linear_source
        else:
            self.linear_target = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        self.attention_scores = nn.Linear(self.n_hidden, 1, bias=False)
        self.attention_activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h: torch.Tensor, adjacency_matrix: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        batch, n_nodes, h_dim = h.shape
        # for each head. We do two linear transformations and then split it up for each head.
        g_source = self.linear_source(h).view(-1, n_nodes, self.n_heads, self.n_hidden)
        g_target = self.linear_target(h).view(-1, n_nodes, self.n_heads, self.n_hidden)
        g_source_repeat = g_source.repeat(n_nodes, 1, 1)
        g_target_repeat_interleaved = g_target.repeat_interleave(n_nodes, dim=0)
        g_sum = g_source_repeat + g_target_repeat_interleaved
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        importance_scores = self.attention_scores(self.attention_activation(g_sum)).squeeze(-1)

        assert adjacency_matrix.shape[0] == 1 or adjacency_matrix.shape[0] == n_nodes
        assert adjacency_matrix.shape[1] == 1 or adjacency_matrix.shape[1] == n_nodes
        assert adjacency_matrix.shape[2] == 1 or adjacency_matrix.shape[2] == self.n_heads

        # e_{ij} is the attention score (importance) from node j to node i
        importance_scores = importance_scores.masked_fill(adjacency_matrix == 0, float("-inf"))
        attention = self.softmax(importance_scores)
        attention_masked = self.dropout_layer(attention)

        head_output = torch.einsum("ijh,jhf->ihf", attention_masked, g_target)
        if self.is_concat:
            return head_output.reshape(n_nodes, self.num_heads * self.n_hidden), attention
        else:
            return head_output.mean(dim=1), attention


class AsynchronousGraphGenerator(nn.Module):
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
            query_includes_categorical: bool = False,
            categorical_input: Optional[int] = None,
            query_includes_type: bool = True,
            transfer_learning: bool = False,
            **kwargs: Any,
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
        self.query_includes_type = query_includes_type
        if query_includes_categorical:
            self.query_includes_categorical = True
            if query_includes_type:
                self.query_dim = (
                        time_embedding_dim
                        + type_embedding_dim
                        + spatial_embedding_dim
                        + categorical_embedding_dim
                )
            else:
                self.query_dim = (
                        time_embedding_dim + spatial_embedding_dim + categorical_embedding_dim
                )
        else:
            self.query_includes_categorical = False
            if query_includes_type:
                self.query_dim = time_embedding_dim + type_embedding_dim + spatial_embedding_dim
            else:
                self.query_dim = time_embedding_dim + spatial_embedding_dim
        self.time_embed = Time2Vec(time_embedding_dim)
        if type_embedding_dim == 0:
            self.type_embed = None
        else:
            self.type_embed = nn.Embedding(num_node_types, type_embedding_dim)
        if spatial_embedding_dim == 0:
            self.spatial_embed = None
        else:
            self.spatial_embed = nn.Embedding(num_spatial_components, spatial_embedding_dim)
        self.num_categorical = num_categories
        if self.query_includes_categorical:
            if num_categories == -1:
                self.categorical_embedding = nn.Linear(
                    categorical_input, categorical_embedding_dim
                )
            else:
                self.categorical_embedding = nn.Embedding(
                    num_categories, categorical_embedding_dim
                )

        self.graph_self_attention = GraphAttentionLayer(
            in_features=self.node_feature_dim,
            out_features=self.node_feature_dim,
            n_heads=num_heads,
            dropout=attention_drop,
            share_weights=True,
            is_concat=False,
        )
        self.elu = nn.ELU()
        # self.node_type_embedding.weight.data.uniform_(-1.0, 1.0)
        # self.spatial_embedding.weight.data.uniform_(-1.0, 1.0)
        # self.category_embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, graph: ContinuousTimeGraphSample, device: torch.device = "cpu") -> [torch.Tensor, torch.Tensor]:
        if len(graph.node_features.shape) < 3:
            features = self.feature_projection(
                graph.node_features.unsqueeze(-1).to(device)
            )
        else:
            features = self.feature_projection(graph.node_features.to(device))
        time_encode = self.time_embed(graph.time.unsqueeze(-1).to(device))
        source_list = [features, time_encode]
        if self.type_embed is not None:
            type_encode = self.type_embed(graph.type_index.to(device))
            source_list.append(type_encode)
        if self.spatial_embed is not None:
            spatial_encode = self.spatial_embed(graph.spatial_index.to(device))
            source_list.append(spatial_encode)
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
        ]
        if self.spatial_embed is not None:
            target_spatial_encode = self.spatial_embed(graph.target.spatial_index.to(device))
            query_list.append(target_spatial_encode)
        if self.query_includes_type and isinstance(
                graph.target.type_index, (torch.LongTensor, torch.FloatTensor)
        ):
            type_encode = self.type_embed(graph.target.type_index.to(device))
            query_list.append(type_encode)
        if self.query_includes_categorical and isinstance(
                graph.target.category_index, (torch.LongTensor, torch.FloatTensor)
        ):
            if len(graph.target.category_index.shape) == 1:
                category = self.categorical_embedding(
                    graph.target.category_index.unsqueeze(-1).to(device)
                )
            else:
                category = self.categorical_embedding(
                    graph.target.category_index.unsqueeze(-2).to(device)
                )
            query_list.append(category)
        target = torch.cat(
            query_list,
            dim=-1,
        )
        key_padding_mask = graph.key_padding_mask.to(device)
        attn_mask = graph.attention_mask.to(device)
        hidden = source
        if torch.any(torch.isnan(source)):
            print(source)
        if torch.any(torch.isnan(target)):
            print(target)
        total_attention = []
        hidden, attention_weights = self.graph_self_attention(hidden, attn_mask)
        y_hat = self.elu(hidden)
        return y_hat, total_attention


if __name__ == "__main__":
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import yaml
    from pathlib import Path

    from Datasets.DoublePendulum.datareader import DoublePendulumDataset
    from AGG.extended_typing import collate_graph_samples

    config_file = "../double_pendulum_config.yaml"
    with open(config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config["data_params"]["batch_size"] = 2
    config["data_params"]["batch_size"] = 10

    train_reader = DoublePendulumDataset(
        db_config=Path("../" + config["data_params"]["db_config"]),
        data_params=config["data_params"],
        create_preprocessing=False,
        version="train",
    )
    train_dataloader = DataLoader(
        train_reader,
        shuffle=False,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
        persistent_workers=False,
    )
    config["model_params"]["num_node_types"] = len(train_reader.type_index)
    config["model_params"]["num_spatial_components"] = len(train_reader.spatial_index)
    config["model_params"]["num_categories"] = len(train_reader.category_index)

    agg = AsynchronousGraphGenerator(**config["model_params"])

    for batch in train_dataloader:
        y_hat, attention_list = agg(batch)
        print(y_hat)
        print(attention_list)
        break
    pass
