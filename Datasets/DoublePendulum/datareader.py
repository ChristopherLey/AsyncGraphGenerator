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
    "mean": [0, 0, 1, 1],
    "std": [0, 0, 0, 0],
    "time": 32705408.0,
    "type": "Normal",
}


def generate_simulation_data(idx, block_size, sparsity, dynamics_params, generation_params):
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
    input_data = data[remainder, :]
    input_categories = categories[remainder, :]
    input_time = np.tile(time[remainder], (4, 1))
    target_time = np.tile(time[removed], (4, 1))
    target_data = data[removed, :]
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

    samples_per_simulation = generation_params["samples_per_simulation"]//2
    interpolation_mask = np.logical_and(greater_than_input, less_than_prediction_limit)
    interpolation_samples = np.argwhere(interpolation_mask).flatten()
    selected_interpolation_samples = np.random.choice(interpolation_samples, samples_per_simulation, replace=False)
    prediction_mask = np.logical_and(np.logical_not(less_than_interpolation_limit), less_than_prediction_limit)
    prediction_samples = np.argwhere(prediction_mask).flatten()
    selected_prediction_samples = np.random.choice(prediction_samples, samples_per_simulation, replace=False)
    samples = np.concatenate([selected_interpolation_samples, selected_prediction_samples])
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
    return graph_lists, idx


def generate_data(
    db_config: dict, dynamics_params: dict, generation_params: dict, block_size: int, exists_ok=True, sparsity=0.5
):
    mongo_db_client = MongoClient(host=db_config["host"], port=db_config["port"])
    db = mongo_db_client[db_config["base"]]
    assert 0 < sparsity < 1.0
    block_name = f"block_{block_size:02d}_{100 * sparsity}%"
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
    meta = {
        "block_size": block_size,
        "sparsity": sparsity,
        "dynamics_params": dynamics_params,
        "generation_params": generation_params,
    }
    block_db.insert_one(meta)
    idx = 0
    for _ in tqdm(range(0, generation_params["train_length"]//generation_params["samples_per_simulation"])):
        graph_lists, idx = generate_simulation_data(idx, block_size, sparsity, dynamics_params, generation_params)
        train_block.insert_many(graph_lists)
    idx = 0
    for _ in tqdm(range(0, generation_params["test_length"]//generation_params["samples_per_simulation"])):
        graph_lists, idx = generate_simulation_data(idx, block_size, sparsity, dynamics_params, generation_params)
        test_block.insert_many(graph_lists)


class DoublePendulumDataset(GraphDataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        block_size: int,
        sparsity: float,
        db_config: Path,
        dynamics_params: dict,
        generation_params: dict,
        version: str = "train",
        create_preprocessing: bool = False,
        shuffle: bool = False,
        subset: Optional[int] = None,

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
        block_name = f"block_{block_size:02d}_{100 * sparsity}%"
        if block_name not in db.list_collection_names():
            if create_preprocessing:
                print(
                    f"No pre-processing for {block_name=}, this could take a while..."
                )
                generate_data(
                    self.mongo_config,
                    dynamics_params,
                    generation_params,
                    block_size=block_size,
                    sparsity=sparsity,
                    exists_ok=False,
                )
            else:
                raise Exception(f"No preprocessing data available for {block_name=}")
        else:
            print(f"Pre-processing found for {block_name=}")
        self.preprocessing_reference = block_name
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
        block_size=1000,
        sparsity=config['data_params']["sparsity"],
        db_config=Path("./data/mongo_config.yaml"),
        dynamics_params=config['data_params']["dynamics_params"],
        generation_params=config['data_params']["generation_params"],
        version="test",
        create_preprocessing=True,
        shuffle=False,
        subset=None,
    )
    print(len(datareader))
    print(datareader[0])