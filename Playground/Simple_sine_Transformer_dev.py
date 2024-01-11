import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
import torch
from torch import nn
from torch import Tensor
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
from datetime import datetime
from torchmetrics import MeanSquaredError

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset
from AGG.utils import FeedForward
from AGG.utils import Time2Vec
from AGG.transformer_model import SelfAttentionBlock, CrossAttentionBlock
from torch.utils.data import DataLoader
from AGG.extended_typing import collate_graph_samples
from Datasets.data_tools import random_index

LOG = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
experiment = datetime.now().strftime("%d-%m_%H:%M:%S")
print(f'Log for {experiment=}')


class AGG(nn.Module):
    def __init__(self,
                 input_dim: int,
                 feature_dim: int,
                 num_heads: int,
                 time_embedding_dim: int,
                 num_layers: int,
                 attention_drop: float = 0.2,
                 dropout: float = 0.2,
                 ):
        super().__init__()
        self.feature_projection = nn.Linear(input_dim, feature_dim)
        self.node_feature_dim = feature_dim + time_embedding_dim
        self.query_dim = time_embedding_dim
        self.time_embed = Time2Vec(time_embedding_dim)
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

    def forward(self, graph: ContinuousTimeGraphSample, device: torch.device = "cpu") -> Tuple[Tensor, list]:
        features = self.feature_projection(graph.node_features.unsqueeze(-1).to(device))
        time_encode = self.time_embed(graph.time.unsqueeze(-1).to(device))
        source_list = [features, time_encode]
        source = torch.cat(
            source_list,
            dim=-1,
        )
        query_list = [
            self.time_embed(graph.target.time.unsqueeze(-1).to(device)),
        ]
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


class SinusoidDataset(GraphDataset):
    def __init__(self, context_length: int | None = None):
        fs = 1000
        t = np.arange(0, 5, 1 / fs)
        f = 10
        x = np.sin(2 * np.pi * f * t)
        self.x = x
        self.t = t
        removed, remainder = random_index(x.shape[0], 0.95)
        self.training_samples = x[remainder]
        self.training_samples_t = t[remainder]
        _, target_index = random_index(removed.shape[0], 0.95)
        self.target_samples = x[removed[target_index]]
        self.target_samples_t = t[removed[target_index]]
        if context_length is None:
            context_length = self.training_samples.shape[0]
        self.dataset = self.generate_data(context_length)

    def generate_data(self, context_length: int) -> list:
        graph_dataset = []
        for i in trange(0, self.training_samples.shape[0] - context_length + 1):
            time = self.training_samples_t[i: i + context_length]
            tau = (time.max() - time)/5.0
            target_times = self.target_samples_t[
                (self.target_samples_t >= time[0])
                & (self.target_samples_t <= time[-1])
                ]
            target_samples_masked = self.target_samples[
                (self.target_samples_t >= time[0])
                & (self.target_samples_t <= time[-1])
                ]
            for j in trange(target_times.shape[0]):
                target_time = (time.max() - target_times[j])/5.0
                graph_dataset.append({
                    "node_features": self.training_samples[i: i + context_length].tolist(),
                    "time": tau.tolist(),
                    'key_padding_mask': (
                            np.zeros_like(time) != 0
                        ).tolist(),
                    "target": {
                        "features": [target_samples_masked[j].tolist(), ],
                        "time": [target_time.tolist(), ],
                    }
                })
        return graph_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ContinuousTimeGraphSample:
        graph = self.graph_transform(self.dataset[idx])
        return graph


model = AGG(
    input_dim=1,
    feature_dim=4,
    num_heads=2,
    time_embedding_dim=4,
    num_layers=1,
    attention_drop=0.0,
    dropout=0.0,
)
sinusoide_dataset = SinusoidDataset()

sinusoid_train_dataloader = DataLoader(
    sinusoide_dataset,
    batch_size=10,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    collate_fn=collate_graph_samples,
)

sinusoid_val_dataloader = DataLoader(
    sinusoide_dataset,
    batch_size=10,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    collate_fn=collate_graph_samples,
)

print(f"Model summary: {model}")
model = model.to(device)

mse_loss = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
train_RMSE = MeanSquaredError(squared=False).to(device)
val_RMSE = MeanSquaredError(squared=False).to(device)
epochs = 1000
total_train = np.inf
total_val = np.inf
lowest_loss = np.inf
with logging_redirect_tqdm():
    prog_bar = trange(epochs, leave=True)
    for epoch in prog_bar:
        model.train()
        for graph_samples in sinusoid_train_dataloader:
            y_hat, total_attention = model(graph_samples, device)
            loss = mse_loss(y_hat, graph_samples.target.features.to(device))
            rmse = train_RMSE(y_hat, graph_samples.target.features.to(device))
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            prog_bar.set_description(f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}", refresh=True)

        with torch.no_grad():
            total_train = train_RMSE.compute()
            prog_bar.set_description(f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}", refresh=True)
            model.eval()
            for graph_samples in sinusoid_val_dataloader:
                y_hat, total_attention = model(graph_samples, device)
                loss = mse_loss(y_hat, graph_samples.target.features.to(device))
                rmse = val_RMSE(y_hat, graph_samples.target.features.to(device))
            total_val = val_RMSE.compute()
            prog_bar.set_description(f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}",
                                     refresh=True)
            if total_val < lowest_loss:
                torch.save(
                    model.state_dict(), f"./best_model_{experiment}.mdl"
                )
                lowest_loss = total_val

