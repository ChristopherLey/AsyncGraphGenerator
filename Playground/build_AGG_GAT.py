from pathlib import Path
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import yaml
from datetime import datetime
from torchmetrics import MeanSquaredError

import matplotlib.pyplot as plt
import numpy as np
from AGG.utils import Time2Vec
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from Datasets.data_tools import random_index


class SinusoidData(object):
    def __init__(self, fs=1000, f=10, t_end=5):
        t = np.arange(0, t_end, 1 / fs)
        x = np.sin(2 * np.pi * f * t)
        self.fs = fs
        self.f = f
        self.x = x
        self.t = t
        t = np.arange(0, t_end, 1 / fs)
        x = np.sin(2 * np.pi * f * t)
        self.x = x
        self.t = t
        removed, remainder = random_index(x.shape[0], 0.97)
        self.training_samples = x[remainder]
        self.training_samples_t = t[remainder]
        _, target_index = random_index(removed.shape[0], 0.97)
        self.target_samples = x[removed[target_index]]
        self.target_samples_t = t[removed[target_index]]


class SinusoidDataset(Dataset):
    def __init__(self, sinusoid: SinusoidData):
        self.sinusoid = sinusoid
        self.x = sinusoid.x
        self.t = sinusoid.t
        self.node = torch.tensor(sinusoid.training_samples).float()
        self.node_t = torch.tensor(sinusoid.training_samples_t).float() / sinusoid.t.max()
        self.target = torch.tensor(sinusoid.target_samples).float()
        self.target_t = torch.tensor(sinusoid.target_samples_t).float() / sinusoid.t.max()
        self.data = []
        for j in range(self.target.shape[0]):
            self.data.append({
                'node': self.node,
                'node_t': self.node_t,
                'target': self.target[j],
                'target_t': self.target_t[j]
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GraphAttentionV2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        """
        * `in_features`, $F$, is the number of input features per node
        * `out_features`, $F\'$, is the number of output features per node
        * `n_heads`, $K$, is the number of attention heads
        * `is_concat` whether the multi-head results should be concatenated or averaged
        * `dropout` is the dropout probability
        * `leaky_relu_negative_slope` is the negative slope for leaky relu activation
        * `share_weights` if set to `True`, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        # i.e. to transform the source node embeddings before self-attention
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # If `share_weights` is `True` the same linear layer is used for the target nodes
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """
        * `h`, $\mathbf{h}$ is the input node embeddings of shape `[n_nodes, in_features]`.
        * `adj_mat` is the adjacency matrix of shape `[n_nodes, n_nodes, n_heads]`.
        We use shape `[n_nodes, n_nodes, 1]` since the adjacency is the same for each head.
        Adjacency matrix represent the edges (or connections) among nodes.
        `adj_mat[i][j]` is `True` if there is an edge from node `i` to node `j`.
        """

        # Number of nodes
        n_nodes = h.shape[0]
        # The initial transformations,
        # $$\overrightarrow{{g_l}^k_i} = \mathbf{W_l}^k \overrightarrow{h_i}$$
        # $$\overrightarrow{{g_r}^k_i} = \mathbf{W_r}^k \overrightarrow{h_i}$$
        # for each head.
        # We do two linear transformations and then split it up for each head.
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score
        #
        # We calculate these for each head $k$. *We have omitted $\cdot^k$ for simplicity*.
        #
        # $$e_{ij} = a(\mathbf{W_l} \overrightarrow{h_i}, \mathbf{W_r} \overrightarrow{h_j}) =
        # a(\overrightarrow{{g_l}_i}, \overrightarrow{{g_r}_j})$$
        #
        # $e_{ij}$ is the attention score (importance) from node $j$ to node $i$.
        # We calculate this for each head.
        #
        # $a$ is the attention mechanism, that calculates the attention score.
        # The paper sums
        # $\overrightarrow{{g_l}_i}$, $\overrightarrow{{g_r}_j}$
        # followed by a $\text{LeakyReLU}$
        # and does a linear transformation with a weight vector $\mathbf{a} \in \mathbb{R}^{F'}$
        #
        #
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # Note: The paper desrcibes $e_{ij}$ as
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big( \mathbf{W}
        # \Big[
        # \overrightarrow{h_i} \Vert \overrightarrow{h_j}
        # \Big] \Big)$$
        # which is equivalent to the definition we use here.

        # First we calculate
        # $\Big[\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j} \Big]$
        # for all pairs of $i, j$.
        #
        # `g_l_repeat` gets
        # $$\{\overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N},
        # \overrightarrow{{g_l}_1}, \overrightarrow{{g_l}_2}, \dots, \overrightarrow{{g_l}_N}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        # `g_r_repeat_interleave` gets
        # $$\{\overrightarrow{{g_r}_1}, \overrightarrow{{g_r}_1}, \dots, \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_r}_2}, \overrightarrow{{g_r}_2}, \dots, \overrightarrow{{g_r}_2}, ...\}$$
        # where each node embedding is repeated `n_nodes` times.
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        # Now we add the two tensors to get
        # $$\{\overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_1} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_1}  +\overrightarrow{{g_r}_N},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_1},
        # \overrightarrow{{g_l}_2} + \overrightarrow{{g_r}_2},
        # \dots, \overrightarrow{{g_l}_2}  + \overrightarrow{{g_r}_N}, ...\}$$
        g_sum = g_l_repeat + g_r_repeat_interleave
        # Reshape so that `g_sum[i, j]` is $\overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}$
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # Calculate
        # $$e_{ij} = \mathbf{a}^\top \text{LeakyReLU} \Big(
        # \Big[
        # \overrightarrow{{g_l}_i} + \overrightarrow{{g_r}_j}
        # \Big] \Big)$$
        # `e` is of shape `[n_nodes, n_nodes, n_heads, 1]`
        e = self.attn(self.activation(g_sum))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # The adjacency matrix should have shape
        # `[n_nodes, n_nodes, n_heads]` or`[n_nodes, n_nodes, 1]`
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        # Mask $e_{ij}$ based on adjacency matrix.
        # $e_{ij}$ is set to $- \infty$ if there is no edge from $i$ to $j$.
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        # We then normalize attention scores (or coefficients)
        # $$\alpha_{ij} = \text{softmax}_j(e_{ij}) =
        # \frac{\exp(e_{ij})}{\sum_{j' \in \mathcal{N}_i} \exp(e_{ij'})}$$
        #
        # where $\mathcal{N}_i$ is the set of nodes connected to $i$.
        #
        # We do this by setting unconnected $e_{ij}$ to $- \infty$ which
        # makes $\exp(e_{ij}) \sim 0$ for unconnected pairs.
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        # $$\overrightarrow{h'^k_i} = \sum_{j \in \mathcal{N}_i} \alpha^k_{ij} \overrightarrow{{g_r}_{j,k}}$$
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)


class GraphAttentionGenerator(nn.Module):
    def __init__(
            self,
            query_dim: int,
            key_dim: int,
            hidden_dim: int,
            dropout: float = 0.1,
            negative_slope: float = 0.2
    ):
        super(GraphAttentionGenerator, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.node_projection = nn.Linear(key_dim, hidden_dim, bias=False)
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.non_linearity = nn.ELU()

    def forward(self, query, key):
        if len(key.shape) == 3:
            # query: [batch_size, 1, query_dim]
            # key: [batch_size, num_nodes, key_dim]
            _, num_nodes, _ = key.shape
            query = query.unsqueeze(1).repeat(1, num_nodes, 1)
        else:
            # query: [1, query_dim]
            # key: [num_nodes, key_dim]
            # query: [num_nodes, query_dim]
            num_nodes = key.shape[0]
            query = query.repeat(num_nodes, 1)
        h_query = self.linear_query(query)
        # h_query: [batch_size, num_nodes, hidden_dim]
        h_key = self.node_projection(key)
        # h_key: [batch_size, num_nodes, hidden_dim]
        h = h_query + h_key
        # h: [batch_size, num_nodes, hidden_dim]
        e = self.leaky_relu(h)
        e = self.attention(e)
        # e: [batch_size, num_nodes, 1]
        e = e.squeeze(-1)
        # e: [batch_size, num_nodes]
        e = self.dropout(e)
        e = torch.softmax(e, dim=-1)
        # e: [batch_size, num_nodes]
        scored_nodes = h_key * e.unsqueeze(-1)
        # scored_value: [batch_size, num_nodes, out_dim]
        if len(key.shape) == 3:
            # scored_nodes: [batch_size, num_nodes, out_dim]
            scored_nodes = torch.sum(scored_nodes, dim=1)
        else:
            # scored_nodes: [num_nodes, out_dim]
            scored_nodes = torch.sum(scored_nodes, dim=0)
        # scored_value: [batch_size, out_dim]
        h_prime = self.non_linearity(scored_nodes)
        return h_prime, e


class GraphSelfAttentionLayer(nn.Module):
    def __init__(
            self,
            node_dim: int,
            hidden_dim: int,
            dropout: float = 0.6,
            negative_slope: float = 0.2,
            share_weights: bool = False
    ):
        super(GraphSelfAttentionLayer, self).__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.share_weights = share_weights
        self.dropout = nn.Dropout(dropout)
        self.linear_l = nn.Linear(node_dim, hidden_dim, bias=False)
        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(node_dim, hidden_dim, bias=False)
        self.attention = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.non_linearity = nn.ELU()

    def forward(self, nodes: torch.Tensor, adj_mat: torch.Tensor):
        # nodes: [batch_size, num_nodes, node_dim]
        batch_size, num_nodes, _ = nodes.shape
        g_left = self.linear_l(nodes)
        # g_left: [batch_size, num_nodes, hidden_dim]
        g_right = self.linear_r(nodes)
        # g_right: [batch_size, num_nodes, hidden_dim]
        g_l_repeat = g_left.repeat(1, num_nodes, 1)
        # g_l_repeat: [batch_size, num_nodes*num_nodes, hidden_dim]
        g_r_repeat_interleave = g_right.repeat_interleave(num_nodes, dim=1)
        # g_r_repeat_interleave: [batch_size, num_nodes*num_nodes, hidden_dim]
        g = g_l_repeat + g_r_repeat_interleave
        # g: [batch_size, num_nodes*num_nodes, hidden_dim]
        g = g.view(batch_size, num_nodes, num_nodes, self.hidden_dim)
        # g: [batch_size, num_nodes, num_nodes, hidden_dim]
        e = self.leaky_relu(g)
        e = self.attention(e)
        # e: [batch_size, num_nodes, num_nodes, 1]
        e = e.squeeze(-1)
        # e: [batch_size, num_nodes, num_nodes]
        a = torch.softmax(e, dim=-1)
        a = self.dropout(a)
        # a: [batch_size, num_nodes, num_nodes]
        attention_result = torch.einsum('bij,bjf->bif', a, g_right)
        # attention_result: [batch_size, num_nodes, hidden_dim]
        attention_result = self.non_linearity(attention_result)
        return attention_result


class AGG(nn.Module):
    def __init__(
            self,
            time_embed_dim: int,
            feature_dim: int,
            feature_projection_dim: int,
            hidden_dim: int,
            out_dim: int,
            dropout: float = 0.1,
            negative_slope: float = 0.2,
            include_linear: bool = True
    ):
        super(AGG, self).__init__()
        self.time2vec = Time2Vec(time_embed_dim, include_linear=include_linear)
        self.key_dim = time_embed_dim + feature_projection_dim
        self.projection = nn.Linear(feature_dim, feature_projection_dim)

        self.graph_attention_generator = GraphAttentionGenerator(
            query_dim=time_embed_dim,
            key_dim=self.key_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            negative_slope=negative_slope
        )
        self.prediction = nn.Linear(hidden_dim, out_dim)

    def forward(self, key_value, key_time, query_time, device: str = 'cpu'):
        key_value = key_value.unsqueeze(-1).to(device)
        key_time = key_time.unsqueeze(-1).to(device)
        query_time = query_time.unsqueeze(-1).to(device)

        key = self.projection(key_value)
        # Time embedding for query and key
        query = self.time2vec(query_time)
        tau = self.time2vec(key_time)

        # Concatenate node_features and time embeddings
        h = torch.cat([key, tau], dim=-1)

        # Generate conditional node
        h_prime, graph_attention = self.graph_attention_generator(query, h)
        y = self.prediction(h_prime)
        return y, graph_attention


sinusoid_data = SinusoidData(f=1, t_end=10)
sinusoid_dataset = SinusoidDataset(sinusoid_data)

experiment = datetime.now().strftime("%d-%m_%H:%M:%S")
print(f'Log for {experiment=}')
dir_path = Path(f'./logs/AGG_{experiment}')
dir_path.mkdir(parents=True, exist_ok=True)

model = AGG(
    time_embed_dim=8,
    feature_dim=1,
    feature_projection_dim=8,
    hidden_dim=32,
    out_dim=1,
    dropout=0.1,
    negative_slope=0.2,
    include_linear=True
)

sinusoid_dataloader = DataLoader(sinusoid_dataset, batch_size=20, shuffle=True, drop_last=False)
LOG = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

print(f"Number of training samples: {len(sinusoid_dataset)}")
print(f"Model summary: {model}")

model = model.to(device)
mse_loss = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
train_RMSE = MeanSquaredError(squared=False).to(device)
val_RMSE = MeanSquaredError(squared=False).to(device)
epochs = 3000
total_train = np.inf
total_val = np.inf
lowest_loss = np.inf

train_loss_plot = []
val_loss_plot = []
val_RMSE_plot = []
train_RMSE_plot = []
t2v_weights = []
t2v_bias = []

with logging_redirect_tqdm():
    prog_bar = trange(epochs, leave=True)
    for epoch in prog_bar:
        model.train()
        for graph_samples in sinusoid_dataloader:
            y_hat, total_attention = model(graph_samples['node'], graph_samples['node_t'], graph_samples['target_t'], device=device)
            if len(graph_samples['target'].shape) == 0:
                graph_samples['target'] = graph_samples['target'].unsqueeze(0)
            loss = mse_loss(y_hat.flatten(), graph_samples['target'].to(device))
            train_loss_plot.append(loss.item())
            t2v_weights.append(model.time2vec.scale.weight.detach().cpu().tolist())
            t2v_bias.append(model.time2vec.scale.bias.detach().cpu().tolist())
            rmse = train_RMSE(y_hat.flatten(), graph_samples['target'].to(device))
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            prog_bar.set_description(
                f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}", refresh=True)

        with torch.no_grad():
            total_train = train_RMSE.compute()
            train_RMSE_plot.append(total_train.item())
            train_RMSE.reset()
            prog_bar.set_description(
                f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}", refresh=True)
            model.eval()
            loss_total = 0
            for graph_samples in sinusoid_dataloader:
                y_hat, total_attention = model(graph_samples['node'], graph_samples['node_t'], graph_samples['target_t'], device=device)
                if len(graph_samples['target'].shape) == 0:
                    graph_samples['target'] = graph_samples['target'].unsqueeze(0)
                loss = mse_loss(y_hat.flatten(), graph_samples['target'].to(device))
                val_loss_plot.append(loss.item())
                loss_total += loss.item()
                rmse = val_RMSE(y_hat.flatten(), graph_samples['target'].to(device))
            total_val = val_RMSE.compute()
            val_RMSE.reset()
            val_RMSE_plot.append(total_val.item())
            prog_bar.set_description(
                f"current loss: {loss.item()}, train_rmse {total_train:.4f}, val_rmse {total_val:.4f}",
                refresh=True)
            if loss_total < lowest_loss:
                torch.save(
                    model.state_dict(), dir_path / f"best_GAT_model_{experiment}.mdl"
                )
                lowest_loss = total_val

plt.figure(figsize=(20, 5))
plt.plot(sinusoid_data.t, sinusoid_data.x, label='sinusoid')
plt.plot(sinusoid_data.training_samples_t, sinusoid_data.training_samples, 'o', label=f'training samples: {sinusoid_data.training_samples.shape[0]}')
plt.plot(sinusoid_data.target_samples_t, sinusoid_data.target_samples, 'o', label=f'target samples: {sinusoid_data.target_samples.shape[0]}')
plt.legend()
plt.savefig(dir_path / 'data_samples.png')

plt.figure(figsize=(20, 5))
plt.plot(train_loss_plot, label='train loss')
plt.plot(val_loss_plot, label='val loss')
plt.xlabel('steps')
plt.legend()
plt.savefig(dir_path / 'loss.png')

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(train_RMSE_plot, label='train RMSE')
plt.plot(val_RMSE_plot, label='val RMSE')
plt.xlabel('epochs')
plt.legend()
plt.subplot(2, 1, 2)
t2v_weights = np.array(t2v_weights)
t2v_bias = np.array(t2v_bias)
for i in range(t2v_weights.shape[1]):
    plt.plot(t2v_weights[:, i], label=f't2v[{i}] weight')
    plt.plot(t2v_bias[:, i], label=f't2v[{i}] bias')
plt.legend()
plt.savefig(dir_path / 'RMSE.png')

with open(dir_path / 'config.yaml', 'w') as f:
    yaml.dump({
        'experiment': experiment,
        'model': model.__class__.__name__,
        'time_embed_dim': model.time2vec.embedding_dim,
        'feature_dim': model.key_dim - model.time2vec.embedding_dim,
        'hidden_dim': model.graph_attention_generator.hidden_dim,
        'out_dim': model.prediction.out_features,
        'dropout': model.graph_attention_generator.dropout.p,
        'negative_slope': model.graph_attention_generator.leaky_relu.negative_slope,
        'include_linear': model.time2vec.include_linear,
        'epochs': epochs,
        'lowest_loss': lowest_loss,
        'f': sinusoid_data.f,
        'fs': sinusoid_data.fs,
    }, f)

tau_query = torch.linspace(0, 1, 1000)
node = sinusoid_dataset.node.unsqueeze(0).repeat(1000, 1)
node_t = sinusoid_dataset.node_t.unsqueeze(0).repeat(1000, 1)

model.load_state_dict(torch.load(dir_path / f"best_GAT_model_{experiment}.mdl"))
model = model.to(device)
model.eval()
with torch.no_grad():
    y_hat, total_attention = model(node, node_t, tau_query, device=device)
    y_hat = y_hat.cpu().numpy()
    total_attention = total_attention.cpu().numpy()

fig, ax = plt.subplots()
plt.imshow(total_attention, extent=[0, 1, 0, 1], cmap='viridis')
ax.axis('tight')
plt.xlabel("node")
plt.ylabel("query")
plt.savefig(dir_path / 'generation_attention.png')

plt.figure(figsize=(20, 5))
plt.plot(tau_query, y_hat, label='generated signal')
plt.plot(sinusoid_data.t/sinusoid_data.t.max(), sinusoid_data.x, label='ground truth signal')
plt.plot(sinusoid_dataset.node_t, sinusoid_dataset.node, 'x', label=f'input')
plt.legend()
plt.savefig(dir_path / 'generation.png')

output = model.time2vec(tau_query.unsqueeze(-1).to(device))
output = output.detach().cpu().numpy()

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
combination = np.sum(output, axis=-1)
pi = r"$\pi$"
for i in range(output.shape[-1]):
    if i == 0 and model.time2vec.include_linear:
        plt.plot(
            np.linspace(0, 1, output.shape[0]),
            output[:, i],
            label=f't2v[{i}], '
                  f'{model.time2vec.scale.weight[i].item():.4f}*t + {model.time2vec.scale.bias[i].item():.2f}')
    else:
        plt.plot(
            np.linspace(0, 1, output.shape[0]),
            output[:, i],
            label=f't2v[{i}], '
                  f'sin(2{pi}*{model.time2vec.scale.weight[i].item():.4f}*t '
                  f'+ {model.time2vec.scale.bias[i].item():.2f})')
plt.legend()
plt.xlabel('t')
plt.subplot(2, 1, 2)
plt.plot(np.linspace(0, 1, output.shape[0]), combination, label='combination')
plt.legend()
plt.xlabel('t')
plt.savefig(dir_path / 't2v.png')
plt.show()
