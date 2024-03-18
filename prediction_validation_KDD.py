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
from torchmetrics import MeanSquaredError
import pickle

from AGG.extended_typing import collate_graph_samples
from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset
from AGG.transformer_model import AsynchronousGraphGeneratorTransformer
from Datasets.Beijing.datareader import create_seq_data_batch
from Datasets.Beijing.datareader import test_masks
from Datasets.Beijing.datareader import unique_stations

sns.set()

figure_path = Path("AGG_diagrams")
model_path = Path("archived_logs/lightning_logs/AGG-kdd_30%_inter-25-08_23:22:17")
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

# Get the data from the database
test_masks = test_masks[0:2]
block_size = config["data_params"]["block_size"]
save_dir = Path("./prediction_data.pkl")
recalculate = True
if save_dir.exists() and not recalculate:
    with open(save_dir, "rb") as f:
        data_set = pickle.load(f)
else:
    data_set = []
    sample_count = 0
    for date_range in tqdm(test_masks):
        raw_data = list(
            db["raw"]
            .find({"time": {"$gte": date_range[0], "$lte": date_range[1]}})
            .sort("time")
        )
        for n in trange(0, len(raw_data) - block_size - 24*15, block_size//2):
            write_data, sample_count, index_datetime, prediction_datetime = create_seq_data_batch(
                raw_data,
                block_size,
                n,
                time_scale,
                params,
                sample_count,
                prediction_steps=45
            )
            if len(write_data) > 0:
                data_set.append({
                    "data": write_data,
                    "index_datetime": index_datetime,
                    "prediction_datetime": prediction_datetime
                })
    with open(save_dir, "wb") as f:
        pickle.dump(data_set, f)

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

time_series: Dict[str, dict] = {
    station: {
        'index_times': [],
        'prediction_times': [],
        'index_values': [],
        'index_types': [],
        'prediction_values': [],
        'target_values': []
    }
    for station in unique_stations
}

batch_size = config["data_params"]["batch_size"]
results = []
lowest_index = 0
lowest_error = 1000000
plot_this_one = 151
with torch.no_grad():
    rmse_metrics = []
    rmse_count = []
    for _ in range(0, 10):
        rmse_metrics.append(MeanSquaredError(squared=False))
        rmse_count.append(0)

    for prediction_block_idx in trange(len(data_set)):
        prediction_block = data_set[prediction_block_idx]
        for l in trange(0, len(prediction_block["data"]), batch_size):
            if l + batch_size > len(prediction_block["data"]):
                graph = create_batch(prediction_block["data"][l:])
            else:
                graph = create_batch(prediction_block["data"][l:l+batch_size])
            y_hat, attention_list = agg(graph, device=device)
            y_hat = y_hat.to("cpu")
            target_feature = graph.target.features
            max_index_time = max(prediction_block['index_datetime'])
            if prediction_block_idx == 0:
                lowest_error = torch.sqrt(torch.mean((y_hat - target_feature) ** 2)).item()
            elif torch.sqrt(torch.mean((y_hat - target_feature) ** 2)).item() < lowest_error:
                lowest_error = torch.sqrt(torch.mean((y_hat - target_feature) ** 2)).item()
                lowest_index = prediction_block_idx
            for n, prediction_time in enumerate(prediction_block['prediction_datetime'][l:l+batch_size]):
                time_index = int((prediction_time - max_index_time).total_seconds()/3600)
                rmse_metrics[time_index](y_hat[n:n+1, 0], target_feature[n:n+1, 0])
                rmse_count[time_index] += 1
            if prediction_block_idx == plot_this_one:
                for m, feature in enumerate(prediction_block['data'][l:l+batch_size]):
                    station = prediction_block['data'][m+l]['target']['spatial_index'][0]
                    time_series[unique_stations[station]]['prediction_times'].append(prediction_block['prediction_datetime'][m+l])
                    time_series[unique_stations[station]]['prediction_values'].append(y_hat[m, 0].item())
                    time_series[unique_stations[station]]['target_values'].append(target_feature[m, 0].item())
        if prediction_block_idx == plot_this_one:
            for m, feature in enumerate(prediction_block['data'][0]['node_features']):
                station = prediction_block['data'][0]['spatial_index'][m]
                index_time = prediction_block['index_datetime'][m]
                time_series[unique_stations[station]]['index_times'].append(index_time)
                time_series[unique_stations[station]]['index_values'].append(feature)
                time_series[unique_stations[station]]['index_types'].append(prediction_block['data'][0]['type_index'][m])
    for n, rmse_metric in enumerate(rmse_metrics):
        if rmse_count[n] == 0:
            continue
        print(f"RMSE for {n} hours: {rmse_metric.compute().item()}, with {rmse_count[n]} samples")
        results.append(rmse_metric.compute().item())
    print(f"Lowest error: {lowest_error} at index {lowest_index}")
del rmse_metrics
colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'lime']
for station in time_series:
    plt.figure(figsize=(20, 10), dpi=300)
    plot_set = set()
    for i in range(len(time_series[station]['index_times'])):
        type = time_series[station]['index_types'][i]
        c = colours[type]
        if type in plot_set:
            plt.plot(time_series[station]['index_times'][i], time_series[station]['index_values'][i], '.', c=c, markersize=10)
        else:
            plot_set.add(type)
            plt.plot(time_series[station]['index_times'][i], time_series[station]['index_values'][i], '.', c=c, label=f'Input Channel:{type}', markersize=10)
    plt.plot(time_series[station]['prediction_times'], time_series[station]['prediction_values'], 's', c=colours[0], label='PM2.5 Prediction', markersize=10)
    plt.plot(time_series[station]['prediction_times'], time_series[station]['target_values'], '*', c='lime', label='PM2.5 Target', markersize=10)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Feature Value", fontsize=18)
    plt.legend()
    plt.title(f"{station} station PM2.5 prediction", fontsize=20)
    plt.savefig(figure_path / f"prediction_station_{station}.png")
    plt.close()


