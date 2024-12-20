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
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader

from AGG.extended_typing import collate_graph_samples
from Datasets.DoublePendulum.datareader import DoublePendulumDataset
from Datasets.DoublePendulum.experiment import ATGDoublePendulumExperiment


model_checkpoint = "artifacts/model-AGG-double_pendulum-29-12_20-25-19:v135/model.ckpt"
config_file = "double_pendulum_config.yaml"

with open(config_file, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

val_reader = DoublePendulumDataset(
    db_config=Path(config["data_params"]["db_config"]),
    data_params=config["data_params"],
    version="test",
)

val_dataloader = DataLoader(
    val_reader,
    shuffle=False,
    batch_size=config["data_params"]["batch_size"],
    drop_last=False,
    num_workers=config["data_params"]["num_workers"],
    collate_fn=collate_graph_samples,
    persistent_workers=False,
)

config["model_params"]["num_node_types"] = len(val_reader.type_index)
config["model_params"]["num_spatial_components"] = len(val_reader.spatial_index)
config["model_params"]["num_categories"] = len(val_reader.category_index)

model = ATGDoublePendulumExperiment(
    model_params=config["model_params"],
    optimiser_params=config["optimiser_params"],
    data_params=config["data_params"],
    logging_params=config["logging_params"],
)

model.load_from_checkpoint(
    model_checkpoint,
    model_params=config["model_params"],
    optimiser_params=config["optimiser_params"],
    data_params=config["data_params"],
    logging_params=config["logging_params"],
)

model.eval()

for batch in val_dataloader:
    loss, y_hat, attention_list = model(batch)
    print(loss)
    break

plt.plot(
    batch.time[:, 0].numpy(), batch.node_features[:, 0].numpy(), "b", label="input"
)
plt.plot(
    batch.target.time[:, 0].numpy(),
    batch.target.features[:, 0].numpy(),
    "x",
    c="g",
    label="ground truth",
)
plt.plot(
    batch.target.time[:, 0].numpy(),
    y_hat[:, 0].detach().numpy(),
    "o",
    c="k",
    label="prediction",
)
plt.legend()
plt.show()
