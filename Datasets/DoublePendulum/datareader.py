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
import copy
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Dict
from typing import Optional

import numpy as np
import yaml
from pymongo import MongoClient
from tqdm import tqdm

from AGG.graph_dataset import GraphDataset
from Datasets.DoublePendulum.dynamics import generate_double_pendulum_data

type_index = [
    "x_1", "y_1", "x_2", "y_2",
]
target_template: Dict[str, list] = {
    "features": [],
    "time": [],
    "type_index": [],
}
graph_template: dict = {
    "node_features": [],
    "key_padding_mask": [],
    "time": [],
    "edge_index": [],
    "type_index": [],
    "attention_mask": [],
}
scaler = {
    "fields": ["x_1", "y_1", "x_2", "y_2"],
    "x_1": {
        'n': 250000000,
        'S_n': -21490.95149520851,
        'S_n^2': 103554996.20868938,
        'mu': -8.596380598083404e-05,
        'std': 0.6435992366721558},
    "y_1": {
        'n': 250000000,
        'S_n': -148201489.7460601,
        'S_n^2': 147707603.53317344,
        'mu': -0.5928059589842404,
        'std': 0.4892969539302987
    },
    "x_2": {
        'n': 250000000,
        'S_n': 34891.564679514166,
        'S_n^2': 236822725.16648498,
        'mu': 0.00013956625871805667,
        'std': 0.9732886936501417
    },
    "y_2": {
        'n': 250000000,
        'S_n': -217286563.0959773,
        'S_n^2': 351887350.23352873,
        'mu': -0.8691462523839092,
        'std': 0.807548260416132
    },
    "time": (0.0, 80.0),
    "type": "Normal",
}


def generate_simulation_data(idx, block_size, sparsity, dynamics_params, generation_params, scale):
    initial_conditions = np.array([
        np.random.uniform(low=-np.pi, high=np.pi),
        np.random.normal(
            generation_params['theta_1_dot_distribution'][0],
            generation_params['theta_1_dot_distribution'][1]),
        np.random.uniform(low=-np.pi, high=np.pi),
        np.random.normal(
            generation_params['theta_2_dot_distribution'][0],
            generation_params['theta_2_dot_distribution'][1])])
    stop_at = int((block_size / (1 - sparsity)) * 1.1 / 4.0)
    data, time = generate_double_pendulum_data(
        initial_conditions,
        stop_at,
        dynamics_params,
        generation_params["sampling_rate"]
    )
    noise = np.random.normal(0, generation_params["signal_std"], size=data.shape)
    data += noise
    mu = np.array([scale['x_1']['mu'], scale['y_1']['mu'], scale['x_2']['mu'], scale['y_2']['mu']]).reshape(1, 4)
    std = np.array([scale['x_1']['std'], scale['y_1']['std'], scale['x_2']['std'], scale['y_2']['std']]).reshape(1, 4)
    normalised_data = (data - mu) / std
    index = np.arange(0, data.shape[0])
    categories = np.tile(np.arange(0, 4), (data.shape[0], 1))
    split_point = np.floor(index.shape[0] * 0.9).astype(int)
    front_index = index[:split_point]
    back_index = index[split_point:]
    np.random.shuffle(front_index)
    remainder = front_index[:block_size // 4]
    removed = front_index[block_size // 4:]
    remainder.sort()
    removed.sort()
    removed = np.concatenate([removed, back_index])
    input_data = normalised_data[remainder, :]
    n = input_data.shape[0]
    S_1 = np.sum(input_data, axis=0)
    S_2 = np.sum(input_data ** 2, axis=0)
    scaler_stats = [n, S_1, S_2]
    scaled_time = (time - scale['time'][0]) / scale['time'][1]
    input_categories = categories[remainder, :]
    input_time = np.tile(scaled_time[remainder], (4, 1))
    target_time = np.tile(scaled_time[removed], (4, 1))
    target_data = normalised_data[removed, :]
    target_categories = categories[removed, :]

    block_input = input_data.flatten()
    block_categories = input_categories.flatten()
    block_time = input_time.T.flatten()
    block_target = target_data.flatten()
    block_target_categories = target_categories.flatten()
    block_target_time = target_time.T.flatten()

    greater_than_input = block_target_time > block_time[0]
    less_than_prediction_limit = block_target_time <= (block_time[-1] + generation_params["prediction_limit"])
    less_than_interpolation_limit = block_target_time <= (block_time[-1])

    if generation_params["prediction_limit"] > 0:
        samples_per_simulation = generation_params["samples_per_simulation"]//2
    else:
        samples_per_simulation = generation_params["samples_per_simulation"]
    interpolation_mask = np.logical_and(greater_than_input, less_than_prediction_limit)
    interpolation_samples = np.argwhere(interpolation_mask).flatten()
    selected_interpolation_samples = np.random.choice(interpolation_samples, samples_per_simulation, replace=False)
    if generation_params["prediction_limit"] > 0:
        prediction_mask = np.logical_and(np.logical_not(less_than_interpolation_limit), less_than_prediction_limit)
        prediction_samples = np.argwhere(prediction_mask).flatten()
        selected_prediction_samples = np.random.choice(prediction_samples, samples_per_simulation, replace=False)
        samples = np.concatenate([selected_interpolation_samples, selected_prediction_samples])
    else:
        samples = selected_interpolation_samples
    samples.sort()

    base_graph = copy.deepcopy(graph_template)
    base_graph["node_features"] = block_input.tolist()
    base_graph["time"] = block_time.tolist()
    base_graph["type_index"] = block_categories.tolist()
    graph_lists = []
    for i in range(samples.shape[0]):
        target = copy.deepcopy(target_template)
        target["features"] = block_target[samples[i]].tolist()
        target["time"] = block_target_time[samples[i]].tolist()
        target["type_index"] = block_target_categories[samples[i]].tolist()
        graph_sample = copy.deepcopy(base_graph)
        graph_sample['target'] = target
        graph_sample['idx'] = idx
        graph_lists.append(graph_sample)
        idx += 1
    return graph_lists, idx, scaler_stats


def generate_data(
    db_config: dict, data_params: dict, exists_ok=True,
):
    mongo_db_client = MongoClient(host=db_config["host"], port=db_config["port"])
    db = mongo_db_client[db_config["base"]]
    assert 0 < data_params["sparsity"] < 1.0
    block_name = data_params["data_set"]
    existing_collections = db.list_collection_names(
        filter={"name": {"$regex": block_name}}
    )
    if not exists_ok:
        for entry in existing_collections:
            db.drop_collection(entry)
    else:
        if len(existing_collections) > 0:
            return
    block_db = db[block_name]
    test_block = block_db["test"]
    train_block = block_db["train"]
    if (
        test_block.estimated_document_count() > 0
        and train_block.estimated_document_count() > 0
    ):
        if exists_ok:
            return
        else:
            train_block.drop()
            test_block.drop()
            block_db.drop()
    meta = data_params
    idx = 0
    generation_params = data_params["generation_params"]
    block_size = data_params["block_size"]
    sparsity = data_params["sparsity"]
    dynamics_params = data_params["dynamics_params"]
    meta['scaler'] = scaler
    scale = [0, np.zeros(4), np.zeros(4)]

    for _ in tqdm(range(0, generation_params["train_length"]//generation_params["samples_per_simulation"])):
        graph_lists, idx, scaler_stats = generate_simulation_data(
            idx, block_size, sparsity, dynamics_params, generation_params, scaler)
        scale[0] += scaler_stats[0]
        scale[1] += scaler_stats[1]
        scale[2] += scaler_stats[2]
        train_block.insert_many(graph_lists)
    idx = 0
    for _ in tqdm(range(0, generation_params["test_length"]//generation_params["samples_per_simulation"])):
        graph_lists, idx, scaler_stats = generate_simulation_data(
            idx, block_size, sparsity, dynamics_params, generation_params, scaler)
        scale[0] += scaler_stats[0]
        scale[1] += scaler_stats[1]
        scale[2] += scaler_stats[2]
        test_block.insert_many(graph_lists)
    mu = scale[1]/scale[0]
    print(np.abs(mu - np.array([scaler['x_1']['mu'], scaler['y_1']['mu'], scaler['x_2']['mu'], scaler['y_2']['mu']])))
    std = np.sqrt((scale[2]/scale[0]) - mu**2)
    print(np.abs(std - np.array([scaler['x_1']['std'], scaler['y_1']['std'], scaler['x_2']['std'], scaler['y_2']['std']])))
    meta['scaler']['sample_stats'] = [scale[0], mu.tolist(), std.tolist()]
    block_db.insert_one(meta)


class DoublePendulumDataset(GraphDataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        db_config: Path,
        data_params: dict,
        version: str = "train",
        create_preprocessing: bool = False,
        shuffle: bool = False,
        subset: Optional[int] = None,
        __overwrite__: bool = False,

    ):
        assert (
            version in self.valid_versions
        ), f"{version=} is not a valid version, valid version include {self.valid_versions}"
        with open(db_config, "r") as f:
            self.mongo_config: dict = yaml.safe_load(f)
        mongo_db_client = MongoClient(
            host=self.mongo_config["host"], port=self.mongo_config["port"]
        )
        db = mongo_db_client[self.mongo_config["base"]]
        data_set = data_params['data_set']
        if data_set not in db.list_collection_names() or __overwrite__:
            if create_preprocessing:
                if __overwrite__:
                    print(f"Overwriting {data_set=}")
                else:
                    print(
                        f"No pre-processing for {data_set=}, this could take a while..."
                    )
                generate_data(
                    self.mongo_config,
                    data_params,
                    exists_ok=False,
                )
            else:
                raise Exception(f"No preprocessing data available for {data_set=}")
        else:
            print(f"Pre-processing found for {data_set=}")
        self.preprocessing_reference = data_set
        self.split = version
        db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ]
        self.meta_data = db_handle.find_one({})
        db_split = db_handle[self.split]
        print(f"Creating index for {db_split}...")
        db_split.create_index("idx")
        print("Done!")
        self.length = db_split.estimated_document_count()
        self.type_index = type_index
        self.spatial_index = []
        self.category_index = []
        self.lazy_loaded = False
        self.db_handle = None
        self.shuffle = shuffle
        self.subset: Optional[int] = None
        if isinstance(subset, int) and 0 < subset < self.length:
            self.subset = subset
        elif subset is not None:
            if isinstance(subset, int):
                Warning(
                    f"subset key should be an integer 0 < subset < dataset_length, instead: {subset=}"
                )
            else:
                Warning(
                    f"subset key should be Union[None, int], instead type: {type(subset)} was found"
                )

    def lazy_load_db(self):
        mongo_db_client = MongoClient(
            host=self.mongo_config["host"], port=self.mongo_config["port"]
        )
        self.db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ][self.split]
        self.lazy_loaded = True

    def __len__(self):
        if self.subset is None:
            return self.length
        else:
            return self.subset

    def __getitem__(self, item):
        if not self.lazy_loaded:
            self.lazy_load_db()
        if self.shuffle:
            item = randint(0, self.length - 1)
        sample = self.db_handle.find_one({"idx": item})
        sample['key_padding_mask'] = [False]*len(sample['node_features'])
        graph = self.graph_transform(sample)
        return graph


if __name__ == "__main__":
    with open("../../double_pendulum_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    datareader = DoublePendulumDataset(
        db_config=Path("./data/mongo_config.yaml"),
        data_params=config["data_params"],
        version="test",
        create_preprocessing=True,
        shuffle=False,
        subset=None,
        __overwrite__=True,
    )
    print(len(datareader))
    print(datareader[0])