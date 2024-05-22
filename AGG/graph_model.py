from typing import Any
from typing import Mapping
from typing import Optional
from typing import Tuple

import torch
from torch import nn

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.utils import FeedForward
from AGG.utils import Time2Vec


class ConditionalGraphAttentionGeneration(nn.Module):
    def __init__(
            self,
            query_dim: int,
            key_dim: int,
            val_dim: int,
            hidden_dim: int,
            num_heads: int = 2,
            dropout: float = 0.1,
            negative_slope: float = 0.2,
            is_concat: bool = True,
            shared_weights: bool = True
    ):
        super(ConditionalGraphAttentionGeneration, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.is_concat = is_concat
        self.dropout = nn.Dropout(dropout)

        if is_concat:
            assert hidden_dim % num_heads == 0
            self.hidden_dim = hidden_dim // num_heads
        else:
            self.hidden_dim = hidden_dim

        self.linear_query = nn.Linear(query_dim, self.hidden_dim * num_heads)
        if shared_weights:
            self.linear_key = self.linear_query
        else:
            self.linear_key = nn.Linear(val_dim, self.hidden_dim * num_heads)

        self.node_projection = nn.Linear(val_dim, self.hidden_dim * num_heads)
        self.attention = nn.Linear(self.hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.non_linearity = nn.ReLU()

    def forward(self, query, key, value):
        # key: [batch_size, num_nodes, key_dim]
        # value: [batch_size, num_nodes, value_dim]
        batch_size, num_nodes, _ = key.shape
        if len(query.shape) != 3:
            # query: [batch_size, query_dim]
            query = query.unsqueeze(1)
        # query: [batch_size, num_nodes, query_dim]
        h_query = self.linear_query(query)
        # h_query: [batch_size, num_nodes, hidden_dim]
        h_key = self.linear_key(key)
        # h_key: [batch_size, num_nodes, hidden_dim]
        h_value = self.node_projection(value)
        # h_value: [batch_size, num_nodes, hidden_dim]

        # Create Heads
        h_query = h_query.view(batch_size, 1, self.num_heads, self.hidden_dim)
        # h_query: [batch_size, 1, num_heads, hidden_dim]
        h_key = h_key.view(batch_size, num_nodes, self.num_heads, self.hidden_dim)
        # h_key: [batch_size, num_nodes, num_heads, hidden_dim]
        h_value = h_value.view(batch_size, num_nodes, self.num_heads, self.hidden_dim)
        # h_value: [batch_size, num_nodes, num_heads, hidden_dim]

        # Duplicate query
        h_query = h_query.repeat(1, num_nodes, 1, 1)
        # h_query: [batch_size, num_nodes, num_heads, hidden_dim]

        # Add query and key
        h = h_query + h_key
        # h: [batch_size, num_nodes, num_heads, hidden_dim]

        e = self.leaky_relu(h)
        e = self.attention(e)
        # e: [batch_size, num_nodes, num_heads, 1]
        e = e.squeeze(-1)
        # e: [batch_size, num_nodes, num_heads]
        e = self.dropout(e)
        e = torch.softmax(e, dim=1)
        # e: [batch_size, num_nodes, num_heads]
        # scored_nodes = torch.einsum("bijh,bjhf->bihf", e, h_value)
        scored_nodes = torch.einsum("bjh,bjhf->bhf", e, h_value)
        # scored_value: [batch_size, num_heads, hidden_dim]
        if self.is_concat:
            scored_nodes = scored_nodes.reshape(batch_size, -1)
        else:
            scored_nodes = torch.mean(scored_nodes, dim=1)
        h_prime = self.non_linearity(scored_nodes)
        return h_prime, e


class GraphAttentionLayer(nn.Module):
    def __init__(
            self,
            query_dim: int,
            key_dim: int,
            val_dim: int,
            hidden_dim: int,
            num_heads: int = 2,
            dropout: float = 0.1,
            negative_slope: float = 0.2,
            is_concat: bool = True,
            share_weights: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.is_concat = is_concat
        self.dropout = nn.Dropout(dropout)

        if is_concat:
            assert hidden_dim % num_heads == 0
            self.hidden_dim = hidden_dim // num_heads
        else:
            self.hidden_dim = hidden_dim

        self.query_projection = nn.Linear(query_dim, self.hidden_dim * num_heads)
        if share_weights:
            self.key_projection = self.query_projection
        else:
            self.key_projection = nn.Linear(key_dim, self.hidden_dim * num_heads)
        self.value_projection = nn.Linear(val_dim, self.hidden_dim * num_heads)
        self.attention = nn.Linear(self.hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.non_linearity = nn.ReLU()

    def forward(self, query, key, value):
        # key: [batch_size, num_nodes, key_dim]
        batch_size, num_nodes, _ = key.shape
        # query: [batch_size, num_nodes, query_dim]
        h_query = self.query_projection(query)
        # h_query: [batch_size, num_nodes, hidden_dim]
        h_key = self.key_projection(key)
        # h_key: [batch_size, num_nodes, hidden_dim]
        h_value = self.value_projection(value)
        # h_value: [batch_size, num_nodes, hidden_dim]

        # Create Heads
        h_query = h_query.view(batch_size, num_nodes, self.num_heads, self.hidden_dim)
        # h_query: [batch_size, num_nodes, num_heads, hidden_dim]
        h_key = h_key.view(batch_size, num_nodes, self.num_heads, self.hidden_dim)
        # h_key: [batch_size, num_nodes, num_heads, hidden_dim]
        h_value = h_value.view(batch_size, num_nodes, self.num_heads, self.hidden_dim)
        # h_value: [batch_size, num_nodes, num_heads, hidden_dim]

        # Duplicate query
        h_query_repeated = h_query.repeat(1, num_nodes, 1, 1)
        # h_query: [batch_size, num_nodes*num_nodes, num_heads, hidden_dim]
        # Duplicate key
        h_key_repeated = h_key.repeat_interleave(num_nodes, dim=1)
        # h_key: [batch_size, num_nodes*num_nodes, num_heads, hidden_dim]

        # Add query and key
        h = h_query_repeated + h_key_repeated
        # h: [batch_size, num_nodes*num_nodes, num_heads, hidden_dim]
        h = h.view(batch_size, num_nodes, num_nodes, self.num_heads, self.hidden_dim)
        # h: [batch_size, num_nodes, num_nodes, num_heads, hidden_dim]

        e = self.leaky_relu(h)
        e = self.attention(e)
        # e: [batch_size, num_nodes, num_nodes, num_heads, 1]
        e = e.squeeze(-1)
        # e: [batch_size, num_nodes, num_nodes, num_heads]
        e = torch.softmax(e, dim=2)
        e = self.dropout(e)
        # e: [batch_size, num_nodes, num_nodes, num_heads]
        scored_nodes = torch.einsum("bijh,bjhf->bihf", e, h_value)
        # scored_value: [batch_size, num_nodes, num_heads, hidden_dim]
        if self.is_concat:
            scored_nodes = scored_nodes.reshape(batch_size, num_nodes, -1)
        else:
            scored_nodes = torch.mean(scored_nodes, dim=2)
        h_prime = self.non_linearity(scored_nodes)
        return h_prime, e


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
            combined_features: bool = False,
            query_includes_categorical: bool = False,
            categorical_input: Optional[int] = None,
            spatial_input: Optional[int] = None,
            query_includes_type: bool = True,
            transfer_learning: bool = False,
            **kwargs: Any,
    ):
        super().__init__()
        self.combined_features = combined_features
        if combined_features:
            self.node_feature_dim = (
                    feature_dim +
                    time_embedding_dim +
                    type_embedding_dim +
                    spatial_embedding_dim +
                    categorical_embedding_dim
            )
        else:
            self.node_feature_dim = feature_dim
        self.feature_projection = nn.Linear(input_dim, feature_dim)
        self.query_includes_type = query_includes_type
        if query_includes_categorical:
            self.query_includes_categorical = True
            if query_includes_type:
                self.meta_dim = (
                        time_embedding_dim
                        + type_embedding_dim
                        + spatial_embedding_dim
                        + categorical_embedding_dim
                )
            else:
                self.meta_dim = (
                        time_embedding_dim + spatial_embedding_dim + categorical_embedding_dim
                )
        else:
            self.query_includes_categorical = False
            if query_includes_type:
                self.meta_dim = time_embedding_dim + type_embedding_dim + spatial_embedding_dim
            else:
                self.meta_dim = time_embedding_dim + spatial_embedding_dim
        self.time_embed = Time2Vec(time_embedding_dim)
        if type_embedding_dim == 0:
            self.type_embed = None
        else:
            self.type_embed = nn.Embedding(num_node_types, type_embedding_dim)
        if spatial_embedding_dim == 0:
            self.spatial_embed = None
        elif num_spatial_components == -1:
            self.spatial_embed = nn.Linear(spatial_input, spatial_embedding_dim)
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
        self.agg_layers = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        if self.combined_features:
            query_dim = self.node_feature_dim
        else:
            query_dim = self.meta_dim
        hidden_dim = query_dim * 2
        for i in range(num_layers):
            if i == 0:
                self.agg_layers.append(
                    GraphAttentionLayer(
                        query_dim=query_dim,
                        key_dim=query_dim,
                        val_dim=self.node_feature_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=attention_drop,
                        share_weights=True,
                        is_concat=True,
                    )
                )
                self.ff_layers.append(
                    nn.Sequential(nn.LayerNorm(hidden_dim),
                                  FeedForward(
                                      input_size=hidden_dim,
                                      hidden_dim=hidden_dim*2,
                                      output_size=hidden_dim,
                                      dropout=dropout,
                                  ),
                                  nn.LayerNorm(hidden_dim),
                                  )
                )
            else:
                if self.combined_features:
                    self.agg_layers.append(
                        GraphAttentionLayer(
                            query_dim=hidden_dim,
                            key_dim=hidden_dim,
                            val_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=attention_drop,
                            share_weights=True,
                            is_concat=True,
                        )
                    )
                else:
                    self.agg_layers.append(
                        GraphAttentionLayer(
                            query_dim=self.meta_dim,
                            key_dim=self.meta_dim,
                            val_dim=hidden_dim,
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            dropout=attention_drop,
                            share_weights=True,
                            is_concat=True,
                        )
                    )
                self.ff_layers.append(
                    nn.Sequential(nn.LayerNorm(hidden_dim),
                                  FeedForward(
                                      input_size=hidden_dim,
                                      hidden_dim=hidden_dim * 2,
                                      output_size=hidden_dim,
                                      dropout=dropout,
                                  ),
                                  nn.LayerNorm(hidden_dim),
                                  )
                )
        if self.combined_features:
            self.generator = ConditionalGraphAttentionGeneration(
                query_dim=self.meta_dim,
                key_dim=hidden_dim,
                val_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attention_drop,
                is_concat=True,
                shared_weights=False,
            )
        else:
            self.generator = ConditionalGraphAttentionGeneration(
                query_dim=self.meta_dim,
                key_dim=self.meta_dim,
                val_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attention_drop,
                is_concat=True,
                shared_weights=True,
            )
        self.head = FeedForward(
                        input_size=hidden_dim,
                        hidden_dim=hidden_dim*2,
                        output_size=input_dim,
                        dropout=dropout,
                    )
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
        meta_data_list = [time_encode, ]
        if self.type_embed is not None:
            type_encode = self.type_embed(graph.type_index.to(device))
            meta_data_list.append(type_encode)
        if self.spatial_embed is not None:
            spatial_encode = self.spatial_embed(graph.spatial_index.to(device))
            meta_data_list.append(spatial_encode)
        if graph.category_index is not None:
            categorical_encode = self.categorical_embedding(
                graph.category_index.to(device)
            )
            meta_data_list.append(categorical_encode)
        meta_data = torch.cat(
            meta_data_list,
            dim=-1,
        )
        if self.combined_features:
            source = torch.cat(
                [features, meta_data],
                dim=-1,
            )
        else:
            source = features
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
        for graph_layer, ff_layer in zip(self.agg_layers, self.ff_layers):
            if self.combined_features:
                hidden, attention_weights = graph_layer(hidden, hidden, hidden)
            else:
                hidden, attention_weights = graph_layer(meta_data, meta_data, hidden)
            total_attention.append(attention_weights)
            hidden = ff_layer(hidden)
        if self.combined_features:
            y_hat, attention_weights = self.generator(target, hidden, hidden)
        else:
            y_hat, attention_weights = self.generator(target, meta_data, hidden)
        total_attention.append(attention_weights)
        y_hat = self.head(y_hat)
        return y_hat, total_attention


if __name__ == "__main__":
    import os
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import yaml
    from pathlib import Path

    from AGG.experiments import AGGExperimentAQIInterpolation
    from AGG.extended_typing import collate_graph_samples
    from Datasets.GRIN_Data.datareader import AQIInterpolationDataset

    config_file = "../aqi_config.yaml"
    with open(config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config["data_params"]["batch_size"] = 2
    subset = None
    shuffle = False

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config["data_params"]["num_workers"] = 0
    persistent_workers = False

    train_reader = AQIInterpolationDataset(
        block_size=config["data_params"]["block_size"],
        sparsity=config["data_params"]["sparsity"],
        block_steps_percent=config["data_params"]["block_steps_percent"],
        db_config=(".." / Path(config["data_params"]["db_config"])),
        dataset=config["data_params"]["dataset"],
        version="train",
        create_preprocessing=True,
    )
    train_dataloader = DataLoader(
        train_reader,
        shuffle=False,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
        persistent_workers=persistent_workers,
    )
    config["model_params"]["num_spatial_components"] = len(train_reader.config["stations"])
    config['logging_params']['scaling'] = train_reader.config['preprocessing']

    agg = AsynchronousGraphGenerator(**config["model_params"])

    for batch in train_dataloader:
        y_hat, attention_list = agg(batch)
        print(y_hat)
        print(attention_list)
        break
    pass
