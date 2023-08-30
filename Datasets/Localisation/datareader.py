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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from tqdm import trange

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset

sequence_keys = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "C01",
    "C02",
    "C03",
    "C04",
    "C05",
    "D01",
    "D02",
    "D03",
    "D04",
    "D05",
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
]
tag_identifiers = [
    "010-000-024-033",
    "020-000-033-111",
    "020-000-032-221",
    "010-000-030-096",
]
activity = [
    "walking",
    "falling",
    "lying down",
    "lying",
    "sitting down",
    "sitting",
    "standing up from lying",
    "on all fours",
    "sitting on the ground",
    "standing up from sitting",
    "standing up from sitting on the ground",
]
target_template: Dict[str, list] = {
    "features": [],
    "time": [],
    "type_index": [],
    "spatial_index": [],
    "category_index": [],
}
graph_template: dict = {
    "node_features": [],
    "key_padding_mask": [],
    "time": [],
    "edge_index": [],
    "type_index": [],
    "spatial_index": [],
    "attention_mask": [],
    "category_index": [],
}
scaler = {
    'fields': ['X', 'Y', 'Z'],
    'mean': [2.8113479961092374, 1.6968769404813713, 0.4182104464547134],
    'std': [0.91622372110056, 0.4737676249131146, 0.3791233780673026],
    'time': 32705408.0,
    'type': 'Normal'
}


def decompose_data(config: dict, block_size: int, exists_ok=True, sparsity=0.5, normal: bool=True):
    with open(config["data_root"], "r") as data_file:
        data_list = data_file.readlines()
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    assert 0 < sparsity < 1.0
    if normal:
        block_name = f"block_{block_size:02d}_{100*sparsity}%_normal"
    else:
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
    parsed_data: dict[str, dict] = {}
    for key in sequence_keys:
        parsed_data[key] = {
            "type_index": [],
            "time": [],
            "datetime": [],
            "node_features": [],
            "category_index": [],
        }
    feature_scaling = []
    for entry in data_list:
        data = (entry.split("\n")[0]).split(",")
        assert data[0] in sequence_keys, f"{data[0]} is an unknown sequence"
        key = data[0]
        parsed_data[key]["type_index"].append(tag_identifiers.index(data[1]))
        parsed_data[key]["time"].append(float(data[2]))
        parsed_data[key]["datetime"].append(
            datetime.strptime(data[3], "%d.%m.%Y %H:%M:%S:%f")
        )
        features = [float(data[4]), float(data[5]), float(data[6])]
        parsed_data[key]["node_features"].append(features)
        feature_scaling.append(features)
        parsed_data[key]["category_index"].append(activity.index(data[7]))
    if normal:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit_transform(feature_scaling)

    block_data = db[block_name]
    test_block = block_data["test"]
    train_block = block_data["train"]
    if normal:
        scaler = {
            "type": "Normal",
            "fields": ["X", "Y", "Z"],
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
            "time": 32705408.0 * sparsity / 0.5,
        }
    else:
        scaler = {
            "type": "MinMax",
            "fields": ["X", "Y", "Z"],
            "min": scaler.data_min_.tolist(),
            "max": scaler.data_max_.tolist(),
            "time": 32705408.0 * sparsity / 0.5,
        }
    meta = {
        "type": block_size,
        "scaler": scaler,
        "sparsity": sparsity,
        "type_index": tag_identifiers,
        "spatial_index": sequence_keys,
        "category_index": activity,
    }
    block_data.insert_one(meta)
    indexes = {}
    for key, value in parsed_data.items():
        idx = np.arange(0, len(parsed_data[key]["time"]))
        np.random.shuffle(idx)
        subset_size = int(np.floor(len(parsed_data[key]["time"]) * sparsity))
        removed = idx[:subset_size]
        remainder = idx[subset_size:]
        removed.sort()
        remainder.sort()
        # removed = np.random.choice(
        #     len(parsed_data[key]["time"]),
        #     size=int(np.floor(len(parsed_data[key]["time"]) * sparsity)),
        #     replace=False,
        # )
        # removed.sort()
        # remainder_idx = np.array(
        #     [i not in removed for i in range(len(parsed_data[key]["time"]))]
        # )
        # remainder = np.arange(len(parsed_data[key]["time"]))[remainder_idx]
        removed_idx = np.arange(removed.shape[0])
        np.random.shuffle(removed_idx)
        test_index = removed_idx[:int(removed.shape[0] // 5)]
        train_index = removed_idx[int(removed.shape[0] // 5):]
        # test_index = np.random.choice(
        #     removed.shape[0], size=removed.shape[0] // 5, replace=False
        # )
        test_index.sort()
        test = removed[test_index]
        # train_index = np.array([i not in test_index for i in range(removed.shape[0])])
        train_index.sort()
        train = removed[train_index]
        indexes[key] = {
            "removed": removed,
            "remainder": remainder,
            "train": train,
            "test": test,
        }
        for input_field in value.keys():
            parsed_data[key][input_field] = np.array(parsed_data[key][input_field])
    max_time = 0
    test_sample_count = 0
    train_sample_count = 0
    for key in tqdm(parsed_data.keys()):
        data_source = parsed_data[key]
        index = indexes[key]
        for n in trange(0, index["remainder"].shape[0] - block_size):
            input_index = index["remainder"][n : n + block_size]
            time = data_source["time"][input_index]
            tau = (time.max() - time) / meta["scaler"]["time"]
            if tau.max() > max_time:
                max_time = tau.max()
            graph = copy.deepcopy(graph_template)
            graph["node_features"] = data_source["node_features"][input_index].tolist()
            graph["key_padding_mask"] = (
                np.zeros_like(data_source["type_index"][input_index]) != 0
            ).tolist()
            graph["time"] = tau.tolist()
            graph["type_index"] = data_source["type_index"][input_index].tolist()
            graph["spatial_index"] = [sequence_keys.index(key)] * block_size
            graph["category_index"] = data_source["category_index"][
                input_index
            ].tolist()
            train_mask = (index["train"] > input_index.min()) & (
                index["train"] < input_index.max()
            )
            test_mask = (index["test"] > input_index.min()) & (
                index["test"] < input_index.max()
            )
            test_set = index["test"][test_mask]
            write_data = []
            for i in range(test_set.shape[0]):
                graph_sample = copy.deepcopy(graph)
                target = copy.deepcopy(target_template)
                target["features"] = [
                    data_source["node_features"][test_set[i]].tolist(),
                ]
                target["type_index"] = [
                    data_source["type_index"][test_set[i]].tolist(),
                ]
                target["spatial_index"] = [
                    sequence_keys.index(key),
                ]
                target["category_index"] = [
                    data_source["category_index"][test_set[i]].tolist(),
                ]
                target["time"] = [
                    (
                        (time.max() - data_source["time"][test_set[i]])
                        / meta["scaler"]["time"]
                    ).tolist(),
                ]
                graph_sample["target"] = target
                graph_sample["idx"] = test_sample_count
                test_sample_count += 1
                write_data.append(graph_sample)
            if len(write_data) > 0:
                test_block.insert_many(write_data)
            train_set = index["train"][train_mask]
            write_data = []
            for i in range(train_set.shape[0]):
                graph_sample = copy.deepcopy(graph)
                target = copy.deepcopy(target_template)
                target["features"] = [
                    data_source["node_features"][train_set[i]].tolist(),
                ]
                target["type_index"] = [
                    data_source["type_index"][train_set[i]].tolist(),
                ]
                target["spatial_index"] = [
                    sequence_keys.index(key),
                ]
                target["category_index"] = [
                    data_source["category_index"][train_set[i]].tolist(),
                ]
                target["time"] = [
                    (
                        (time.max() - data_source["time"][train_set[i]])
                        / meta["scaler"]["time"]
                    ).tolist(),
                ]
                graph_sample["target"] = target
                graph_sample["idx"] = train_sample_count
                train_sample_count += 1
                write_data.append(graph_sample)
            if len(write_data) > 0:
                train_block.insert_many(write_data)


class ActivityData(GraphDataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        block_size: int,
        sparsity: float,
        db_config: Path,
        version: str = "train",
        create_preprocessing: bool = False,
        shuffle: bool = False,
        subset: Optional[int] = None,
        normal: bool = False
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
        if normal:
            block_name = f"block_{block_size:02d}_{100 * sparsity}%_normal"
        else:
            block_name = f"block_{block_size:02d}_{100 * sparsity}%"
        if block_name not in db.list_collection_names():
            if create_preprocessing:
                print(
                    f"No pre-processing for {block_name=}, this could take a while..."
                )
                decompose_data(
                    self.mongo_config,
                    block_size=block_size,
                    sparsity=sparsity,
                    exists_ok=False,
                    normal=normal
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
        self.category_index = self.meta_data.pop("category_index")
        self.spatial_index = self.meta_data.pop("spatial_index")
        self.type_index = self.meta_data.pop("type_index")
        db_split = db_handle[self.split]
        print(f"Creating index for {db_split}...")
        db_split.create_index("idx")
        print("Done!")
        self.length = db_split.estimated_document_count()
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
        sample = self.graph_transform(self.db_handle.find_one({"idx": item}))
        return sample


def test_datareader():
    test_obj = ActivityData(
        block_size=30,
        sparsity=0.5,
        db_config=Path("./data/mongo_config.yaml"),
        version="test",
        create_preprocessing=True,
        normal=True
    )
    print(len(test_obj))  # 1895393
    assert isinstance(len(test_obj), int)
    sample: ContinuousTimeGraphSample = test_obj[1]
    assert not sample.node_features.isnan().any()
    assert not sample.target.features.isnan().any()
