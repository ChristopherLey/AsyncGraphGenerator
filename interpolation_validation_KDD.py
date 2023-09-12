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
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from pymongo import MongoClient
from tqdm import tqdm
from tqdm import trange

from AGG.extended_typing import collate_graph_samples
from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset
from AGG.transformer_model import AsynchronousGraphGeneratorTransformer
from Datasets.Beijing.datareader import create_data_block
from Datasets.Beijing.datareader import features
from Datasets.Beijing.datareader import random_index
from Datasets.Beijing.datareader import test_masks
from Datasets.Beijing.datareader import training_masks
from Datasets.Beijing.datareader import unique_stations

sns.set()

figure_path = Path("AGG_diagrams")
model_path = Path("lightning_logs/AGG-kdd_30%_inter-25-08_23:22:17")
model_config_path = model_path / "config.yaml"
best_checkpoint = (
    model_path / "checkpoints" / "model-epoch=05-val_RMSE_epoch=0.157463.ckpt"
)
checkpoint = torch.load(best_checkpoint)
# Load the config file
with open(model_config_path, "r") as f:
    config: dict = yaml.safe_load(f)
# Load the config file
with open("Datasets/Beijing/data/mongo_config.yaml", "r") as f:
    mongo_config: dict = yaml.safe_load(f)

# Connect to the database
mongo_db_client = MongoClient(host=mongo_config["host"], port=mongo_config["port"])
db = mongo_db_client[mongo_config["base"]]
db_params = db["param"]
params = db_params.find_one({})
time_scale = 3600 * 72
# Get the date range
date_range = (training_masks[0][0], test_masks[2][1])

# Get the data from the database
sparsity = 0.3
block_size = config["data_params"]["block_size"]
raw_data = list(
    db["raw"]
    .find({"time": {"$gte": date_range[0], "$lte": date_range[1]}})
    .sort("time")
)
removed, remainder = random_index(len(raw_data), sparsity)
data_set = []
sample_count = 0
for n in trange(0, remainder.shape[0] - block_size, block_size):
    write_data, sample_count = create_data_block(
        {
            "remainder": remainder,
            "removed": removed,
        },
        raw_data,
        block_size,
        n,
        time_scale,
        True,
        params,
        sample_count,
    )
    if len(write_data) > 0:
        data_set.append(write_data)

config["model_params"].pop("type")
agg = AsynchronousGraphGeneratorTransformer(**config["model_params"])
agg.load_state_dict(checkpoint["state_dict"])
agg.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agg = agg.to(device)


def create_batch(data_set: list) -> ContinuousTimeGraphSample:
    batch = []
    for data in data_set:
        graph = GraphDataset.graph_transform(data)
        batch.append(graph)
    return collate_graph_samples(batch)


def interpolate_mean(
        time_array: np.ndarray,
        value_array: np.ndarray,
        est_time_array: np.ndarray,
        kernel_width: int
) -> np.ndarray:
    interpolated_value_array = np.zeros_like(est_time_array)
    for n, est_time in enumerate(est_time_array):
        index = np.argmin(np.abs(time_array - est_time))

        if index - kernel_width // 2 < 0:
            start = 0
            end = kernel_width
        elif index + kernel_width // 2 > len(time_array):
            start = len(time_array) - kernel_width
            end = len(time_array)
        else:
            start = index - kernel_width // 2
            end = index + kernel_width // 2
        interpolated_value_array[n] = np.mean(value_array[start:end])
    return interpolated_value_array


time_series: Dict[str, dict] = {
    station: {
        "input": [],
        "time": [],
        "pm25_est": [],
        "pm25": [],
        "est_time": [],
    }
    for station in unique_stations
}
max_time = 0
with torch.no_grad():
    for data in tqdm(data_set):
        type_index = np.array(data[0]["type_index"])
        type_bool = type_index == features.index("PM2.5")
        pm25_feature = np.array(data[0]["node_features"])[type_bool]
        pm25_spacial = np.array(data[0]["spatial_index"])[type_bool]
        pm25_time = np.array(data[0]["time"])[type_bool]
        max_time += pm25_time.max()
        pm25_time = max_time - pm25_time

        for i, station in enumerate(unique_stations):
            station_bool = pm25_spacial == i
            time_series[station]["input"].append(pm25_feature[station_bool])
            time_series[station]["time"].append(pm25_time[station_bool])
        if len(data) > config["data_params"]["batch_size"]:
            for n in range(0, len(data), config["data_params"]["batch_size"]):
                last_point = n + config["data_params"]["batch_size"]
                if last_point >= len(data):
                    subset = data[n:last_point]
                else:
                    subset = data[n:last_point]
                graph = create_batch(subset)
                y_hat, attention_list = agg(graph, device=device)
                y_hat = y_hat.to("cpu")
                target_feature = graph.target.features
                target_time = max_time - graph.target.time
                for i, station in enumerate(unique_stations):
                    station_bool = graph.target.spatial_index == i
                    time_series[station]["pm25_est"].append(y_hat[station_bool].numpy())
                    time_series[station]["pm25"].append(
                        target_feature[station_bool].numpy()
                    )
                    time_series[station]["est_time"].append(
                        target_time[station_bool].numpy()
                    )
        else:
            graph = create_batch(data)
            y_hat, attention_list = agg(graph, device=device)
            y_hat = y_hat.to("cpu")
            target_feature = graph.target.features
            target_time = max_time - graph.target.time
            for i, station in enumerate(unique_stations):
                station_bool = graph.target.spatial_index == i
                time_series[station]["pm25_est"].append(y_hat[station_bool].numpy())
                time_series[station]["pm25"].append(
                    target_feature[station_bool].numpy()
                )
                time_series[station]["est_time"].append(
                    target_time[station_bool].numpy()
                )
for station in unique_stations:
    time_series[station]["input"] = np.concatenate(
        time_series[station]["input"], axis=0
    )
    time_series[station]["time"] = np.concatenate(time_series[station]["time"], axis=0)
    time_series[station]["pm25_est"] = np.concatenate(
        time_series[station]["pm25_est"], axis=0
    )
    time_series[station]["pm25"] = np.concatenate(time_series[station]["pm25"], axis=0)
    time_series[station]["est_time"] = np.concatenate(
        time_series[station]["est_time"], axis=0
    )
    interpolation_mean_3 = interpolate_mean(
        time_series[station]["time"],
        time_series[station]["input"],
        time_series[station]["est_time"],
        3)
    interpolation_mean_5 = interpolate_mean(
        time_series[station]["time"],
        time_series[station]["input"],
        time_series[station]["est_time"],
        5)
    interpolation_mean_7 = interpolate_mean(
        time_series[station]["time"],
        time_series[station]["input"],
        time_series[station]["est_time"],
        7)
    interpolation_mean_9 = interpolate_mean(
        time_series[station]["time"],
        time_series[station]["input"],
        time_series[station]["est_time"],
        9)


    plt.figure(figsize=(20, 10), dpi=300)
    plt.plot(
        time_series[station]["time"] * 72,
        time_series[station]["input"],
        "-.",
        c="g",
        lw=3,
        label="Input",
    )
    plt.plot(
        time_series[station]["est_time"] * 72,
        time_series[station]["pm25_est"],
        "o",
        c="r",
        label="Estimate",
    )
    plt.plot(
        time_series[station]["est_time"] * 72,
        time_series[station]["pm25"],
        "*",
        c="k",
        label="Target",
    )
    plt.legend()
    plt.xlabel("Time (h)")
    plt.ylabel("PM2.5")
    plt.title(f"PM2.5 at {station} with sparsity={sparsity*100:2g}%"
              f"\nRMSE: {np.sqrt(np.mean(np.power(time_series[station]['pm25_est'] - time_series[station]['pm25'], 2.0))):3g}"
              f"\nRMSE mean kernel:3: {np.sqrt(np.mean(np.power(interpolation_mean_3 - time_series[station]['pm25'], 2.0))):3g}"
              f"\nRMSE mean kernel:5: {np.sqrt(np.mean(np.power(interpolation_mean_5 - time_series[station]['pm25'], 2.0))):3g}"
              f"\nRMSE mean kernel:7: {np.sqrt(np.mean(np.power(interpolation_mean_7 - time_series[station]['pm25'], 2.0))):3g}"
              f"\nRMSE mean kernel:9: {np.sqrt(np.mean(np.power(interpolation_mean_9 - time_series[station]['pm25'], 2.0))):3g}")
    plt.savefig(figure_path / f"pm25_{station}_{sparsity*100:2g}%.png")
    plt.close()
