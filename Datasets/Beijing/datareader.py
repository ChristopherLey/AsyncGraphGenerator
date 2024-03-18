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
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm import trange

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset
from Datasets.data_tools import random_index

features = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
time = "datetime"
category = "wd"
spatial = "station"
unique_wd = [
    "NNW",
    "E",
    "NW",
    "WNW",
    "N",
    "ENE",
    "NNE",
    "W",
    "NE",
    "SSW",
    "ESE",
    "SE",
    "S",
    "SSE",
    "SW",
    "WSW",
    "None",
]
unique_stations = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]

training_masks = [
    (datetime(year=2014, month=5, day=1), datetime(year=2014, month=6, day=1)),
    (datetime(year=2014, month=7, day=1), datetime(year=2014, month=9, day=1)),
    (datetime(year=2014, month=10, day=1), datetime(year=2014, month=12, day=1)),
    (datetime(year=2015, month=1, day=1), datetime(year=2015, month=3, day=1)),
    (datetime(year=2015, month=4, day=1), datetime(year=2015, month=4, day=30)),
]
test_masks = [
    (datetime(year=2014, month=6, day=1), datetime(year=2014, month=7, day=1)),
    (datetime(year=2014, month=9, day=1), datetime(year=2014, month=10, day=1)),
    (datetime(year=2014, month=12, day=1), datetime(year=2015, month=1, day=1)),
    (datetime(year=2015, month=3, day=1), datetime(year=2015, month=4, day=1)),
]
target_template: Dict[str, list] = {
    "features": [],
    "time": [],
    "type_index": [],
    "spatial_index": [],
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


def split_string(strings: str):
    return {
        "time": datetime(
            year=int(strings[1]),
            month=int(strings[2]),
            day=int(strings[3]),
            hour=int(strings[4]),
        ),
        "PM2.5": strings[5],
        "PM10": strings[6],
        "S02": strings[7],
        "N02": strings[8],
        "CO": strings[9],
        "03": strings[10],
        "TEMP": strings[11],
        "PRES": strings[12],
    }


def decompose_pm2_5_data(file: Path):
    decomposed_data = []
    header = []
    with open(file, "r", encoding="utf-8") as f:
        contents = f.readlines()
        for i, line in enumerate(contents):
            strings = line[:-1].split(",")
            if i == 0:
                for title in strings:
                    title = title.split('"')[1]
                    if title not in ["year", "month", "day", "hour"]:
                        header.append(title)
                    else:
                        if title == "year":
                            header.append("datetime")
            else:
                data: list = []
                idx = 0
                for data_field in header:
                    if data_field == "datetime":
                        idx = 5
                        data.append(
                            datetime(
                                year=int(strings[1]),
                                month=int(strings[2]),
                                day=int(strings[3]),
                                hour=int(strings[4]),
                            )
                        )
                    else:
                        if strings[idx] == "NA":
                            data.append(None)
                        else:
                            if idx in [15, 17]:
                                data.append(strings[idx].split('"')[1])
                            else:
                                data.append(float(strings[idx]))
                        idx += 1
                decomposed_data.append(data)
    return header, decomposed_data


def decompose_KDD(config: dict, exist_ok: bool = True):
    root_path = Path(config["data_root"])
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    db_raw = db["raw"]
    count = db_raw.estimated_document_count()
    if count > 0:
        if exist_ok:
            return
        else:
            db_raw.drop()
    idx = 0
    for data_csv in root_path.iterdir():
        header, decomposed_data = decompose_pm2_5_data(data_csv)
        datasets = []
        for entry_list in decomposed_data:
            for type_index, feature in enumerate(features):
                slice = header.index(feature)
                node_feature = entry_list[slice]
                if node_feature is None:
                    continue
                if entry_list[12] is None:
                    wd = "None"
                else:
                    wd = entry_list[12]
                raw_entry = {
                    "idx": idx,
                    "time": entry_list[1],
                    "node_features": node_feature,
                    "category_index": unique_wd.index(wd),
                    "type_index": type_index,
                    "spatial_index": unique_stations.index(entry_list[14]),
                }
                idx += 1
                datasets.append(raw_entry)
        db_raw.insert_many(datasets)


def compute_normalisation(config: dict, exist_ok: bool = True):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    db_raw = db["raw"]
    count = db_raw.estimated_document_count()
    assert count > 0, f"{db_raw}:{count=}"
    db_param = db["param"]
    if db_param.estimated_document_count() > 0:
        if exist_ok:
            return
        else:
            db_param.drop()
    params: Dict[str, Any] = {
        "scaling": {},
        "features": features,
        "categories": unique_wd,
        "spatial": unique_stations,
    }
    raw_data_lists: Dict[str, list] = {}
    for feature in features:
        feature_cursor = db_raw.find({"type_index": {"$eq": features.index(feature)}})
        raw_data_lists[feature] = []
        for item in feature_cursor:
            raw_data_lists[feature].append(item["node_features"])
    for key, item in raw_data_lists.items():
        params["scaling"][key] = {"mean": np.mean(item), "std": np.std(item)}
    db_param.insert_one(params)


def normalise(feature, scaling):
    return (feature - scaling["mean"]) / scaling["std"]


def create_indexes(
    config: dict, sparsity: float,
):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    db_raw = db["raw"]
    assert db_raw.estimated_document_count() > 0
    raw_train = []
    for mask in training_masks:
        raw_train.append(
            list(db_raw.find({"time": {"$gte": mask[0], "$lte": mask[1]}}).sort("time"))
        )
    raw_test = []
    for mask in test_masks:
        raw_test.append(
            list(db_raw.find({"time": {"$gt": mask[0], "$lt": mask[1]}}).sort("time"))
        )
    train_indexes = []
    for data in raw_train:
        removed, remainder = random_index(len(data), sparsity)
        train_indexes.append(
            {"removed": removed.tolist(), "remainder": remainder.tolist()}
        )
    test_indexes = []
    for data in raw_test:
        removed, remainder = random_index(len(data), sparsity)
        test_indexes.append(
            {"removed": removed.tolist(), "remainder": remainder.tolist()}
        )
    index = {"train": train_indexes, "test": test_indexes}
    return index, raw_train, raw_test


def create_data_block(
    index: dict,
    raw: list,
    block_size: int,
    n: int,
    time_scale: float,
    params,
    sample_count: int,
):
    write_data = []
    input_index = index["remainder"][n : (n + block_size)]
    max_time = raw[input_index[-1]]["time"]
    graph = copy.deepcopy(graph_template)
    for k in range(len(input_index)):
        raw_entry = raw[input_index[k]]
        graph["node_features"].append(
            normalise(
                raw_entry["node_features"],
                params["scaling"][features[raw_entry["type_index"]]],
            )
        )
        graph["time"].append(
            (max_time - raw_entry["time"]).total_seconds() / time_scale
        )
        graph["category_index"].append(raw_entry["category_index"])
        graph["type_index"].append(raw_entry["type_index"])
        graph["spatial_index"].append(raw_entry["spatial_index"])
    graph["key_padding_mask"] = (np.zeros_like(input_index) != 0).tolist()
    removed_index_array = np.array(index["removed"])
    mask = (removed_index_array > min(input_index)) & (
        removed_index_array < max(input_index)
    )
    graph_target = removed_index_array[mask]
    for i in range(graph_target.shape[0]):
        k = graph_target[i]
        if raw[k]["type_index"] == 0:
            graph_sample = copy.deepcopy(graph)
            target = copy.deepcopy(target_template)
            feature_type = features[raw[k]["type_index"]]
            target["features"] = [
                normalise(raw[k]["node_features"], params["scaling"][feature_type]),
            ]
            target["type_index"] = [
                raw[k]["type_index"],
            ]
            target["spatial_index"] = [
                raw[k]["spatial_index"],
            ]
            target["category_index"] = [
                raw[k]["category_index"],
            ]
            target["time"] = [
                (max_time - raw[k]["time"]).total_seconds() / time_scale,
            ]
            graph_sample["target"] = target
            graph_sample["idx"] = sample_count
            sample_count += 1
            write_data.append(graph_sample)
    return write_data, sample_count


def create_seq_data_batch(
    raw: list,
    block_size: int,
    n: int,
    time_scale: float,
    params,
    sample_count: int,
    prediction_steps: int = 24*15,
):
    write_data = []
    input_index = np.arange(n, (n + block_size))
    prediction_indexes = []
    max_times = []
    for i in range(n + block_size, len(raw)):
        if raw[i]["type_index"] == 0:
            prediction_indexes.append(i)
            max_times.append(raw[i]["time"])
            if len(prediction_indexes) == prediction_steps:
                break
    graph = copy.deepcopy(graph_template)
    index_datetime = []
    prediction_datetime = []
    for idx in range(len(input_index)):
        raw_entry = raw[input_index[idx]]
        graph["node_features"].append(
            normalise(
                raw_entry["node_features"],
                params["scaling"][features[raw_entry["type_index"]]],
            )
        )
        graph["time"].append(
            raw_entry["time"]
        )
        index_datetime.append(raw_entry["time"])
        graph["category_index"].append(raw_entry["category_index"])
        graph["type_index"].append(raw_entry["type_index"])
        graph["spatial_index"].append(raw_entry["spatial_index"])
    graph["key_padding_mask"] = (np.zeros_like(input_index) != 0).tolist()
    for idx, max_time in zip(prediction_indexes, max_times):
        graph_sample = copy.deepcopy(graph)
        target = copy.deepcopy(target_template)
        feature_type = features[raw[idx]["type_index"]]
        target["features"] = [
            normalise(raw[idx]["node_features"], params["scaling"][feature_type]),
        ]
        target["type_index"] = [
            raw[idx]["type_index"],
        ]
        target["spatial_index"] = [
            raw[idx]["spatial_index"],
        ]
        target["category_index"] = [
            raw[idx]["category_index"],
        ]
        target["time"] = [
            (max_time - raw[idx]["time"]).total_seconds() / time_scale,
        ]
        prediction_datetime.append(raw[idx]["time"])
        graph_sample["target"] = target
        graph_sample["idx"] = sample_count
        for i in range(len(graph_sample["time"])):
            graph_sample["time"][i] = (max_time - graph_sample["time"][i]).total_seconds() / time_scale
        sample_count += 1
        write_data.append(graph_sample)
    return write_data, sample_count, index_datetime, prediction_datetime


def create_interpolation_dataset(
    config: dict,
    block_size: int,
    exists_ok: bool = True,
    sparsity: float = 0.5,
    block_steps: int = 25,
    time_scale: int = 3600 * 72,
):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    db_params = db["param"]
    assert db_params.estimated_document_count() > 0
    db_raw = db["raw"]
    assert db_raw.estimated_document_count() > 0
    params = db_params.find_one({})
    block_name = f"block_{block_size:02d}_{100 * sparsity}%_steps_{block_steps}"
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
    print(f"Creating interpolation dataset for {block_name}")
    indexes, raw_train, raw_test = create_indexes(config, sparsity)
    test_sample_count = 0
    train_sample_count = 0

    for i in trange(len(raw_train)):
        index = indexes["train"][i]
        raw = raw_train[i]
        for n in trange(0, len(index["remainder"]) - block_size, block_steps):
            write_data, train_sample_count = create_data_block(
                index,
                raw,
                block_size,
                n,
                time_scale,
                params,
                train_sample_count,
            )
            if len(write_data) > 0:
                train_block.insert_many(write_data)
    for i in trange(len(raw_test)):
        index = indexes["test"][i]
        raw = raw_test[i]
        for n in trange(0, len(index["remainder"]) - block_size, block_steps):
            write_data, test_sample_count = create_data_block(
                index,
                raw,
                block_size,
                n,
                time_scale,
                params,
                test_sample_count,
            )
            if len(write_data) > 0:
                test_block.insert_many(write_data)


def decompose_data_regression_block(
    config: dict, block_size: int, exists_ok: bool = True
):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client["Beijing"]
    df: pd.DataFrame = pd.read_hdf(Path(config["data_root"]) / "pm2_5_df.h5")  # noqa
    locations = len(unique_stations)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    block_name = f"block_{block_size:02d}"
    existing_collections = db.list_collection_names(
        filter={"name": {"$regex": block_name}}
    )
    if not exists_ok:
        for entry in existing_collections:
            db.drop_collection(entry)
    else:
        if len(existing_collections) > 0:
            return

    block_data = db[block_name]
    meta = {
        "type": block_size,
        "scaler": {
            "type": "MinMax",
            "fields": features,
            "min": scaler.data_min_.tolist(),
            "max": scaler.data_max_.tolist(),
            "time": block_size * 3600,
        },
        "test_masks": [
            (
                mask[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                mask[1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            for mask in test_masks
        ],
        "train_masks": [
            (
                mask[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                mask[1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            for mask in training_masks
        ],
        "type_index": features,
        "spatial_index": unique_stations,
        "category_index": unique_wd,
    }
    block_data.insert_one(meta)
    for i, mask_set in enumerate([training_masks, test_masks]):
        if i == 0:
            version = "train"
            sample_count = 0
        else:
            version = "test"
            sample_count = 0
        print(f"Generating graphs for {version}ing")
        split = block_data[version]
        for mask in tqdm(mask_set, desc="mask_range"):
            df_slice = df[(df[time] > mask[0]) & (df[time] < mask[1])]
            for k in trange(len(df_slice) // locations - block_size, desc="sample_set"):
                sample = df_slice.iloc[k * locations : (k + block_size) * locations]
                target_time = sample[time].max()
                valid_targets = sample[time] == target_time
                targets = sample[valid_targets]
                sources = sample[~valid_targets]
                graph = copy.deepcopy(graph_template)
                data = []
                for source in range(len(sources)):
                    line = sources.iloc[source]
                    tau = (target_time - line[time]).seconds / (block_size * 3600)
                    for m, key in enumerate(features):
                        if np.isnan(line[key]):
                            graph["key_padding_mask"].append(True)
                            graph["node_features"].append(0)
                        else:
                            graph["node_features"].append(line[key])
                            graph["key_padding_mask"].append(False)
                        graph["type_index"].append(m)
                        graph["spatial_index"].append(
                            unique_stations.index(line[spatial])
                        )
                        graph["category_index"].append(unique_wd.index(line[category]))
                        graph["time"].append(tau)
                for target_entry in range(len(targets)):
                    line = targets.iloc[target_entry]
                    tau = (target_time - line[time]).seconds / (block_size * 3600)
                    for m, key in enumerate(features):
                        target = copy.deepcopy(target_template)
                        if np.isnan(line[key]):  # invalid target feature
                            continue
                        graph_sample = copy.deepcopy(graph)
                        target["features"].append(line[key])
                        target["type_index"].append(m)
                        target["spatial_index"].append(
                            unique_stations.index(line[spatial])
                        )
                        target["time"].append(tau)

                        for n in range(len(targets)):
                            not_targets = targets.iloc[n]
                            tau2 = (target_time - not_targets[time]).seconds / (
                                block_size * 3600
                            )
                            for o, key in enumerate(features):
                                if n == target:
                                    if o != m:
                                        if np.isnan(not_targets[key]):
                                            graph_sample["node_features"].append(0)
                                            graph_sample["key_padding_mask"].append(
                                                True
                                            )
                                        else:
                                            graph_sample["node_features"].append(
                                                not_targets[key]
                                            )
                                            graph_sample["key_padding_mask"].append(
                                                False
                                            )
                                        graph_sample["type_index"].append(o)
                                        graph_sample["spatial_index"].append(
                                            unique_stations.index(not_targets[spatial])
                                        )
                                        graph_sample["category_index"].append(
                                            unique_wd.index(not_targets[category])
                                        )
                                        graph_sample["time"].append(tau2)
                                else:
                                    if np.isnan(not_targets[key]):
                                        graph_sample["node_features"].append(0)
                                        graph_sample["key_padding_mask"].append(True)
                                    else:
                                        graph_sample["node_features"].append(
                                            not_targets[key]
                                        )
                                        graph_sample["key_padding_mask"].append(False)
                                    graph_sample["type_index"].append(o)
                                    graph_sample["spatial_index"].append(
                                        unique_stations.index(not_targets[spatial])
                                    )
                                    graph_sample["category_index"].append(
                                        unique_wd.index(not_targets[category])
                                    )
                                    graph_sample["time"].append(tau2)
                        graph_sample["target"] = target
                        graph_sample["idx"] = sample_count
                        data.append(graph_sample)
                        sample_count += 1
                split.insert_many(data)


class KDDInterpolationDataset(GraphDataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        block_size: int,
        sparsity: float,
        db_config: Path,
        version: str = "train",
        subset: Optional[int] = None,
        shuffle: bool = False,
        block_steps: int = int(3200 // 1.5),
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
        block_name = f"block_{block_size:02d}_{100 * sparsity}%_steps_{block_steps}.{version}"

        if block_name not in db.list_collection_names():
            raise Exception(f"No preprocessing data available for {block_name=}")
        else:
            print(f"Pre-processing found for {block_name=}")
        self.preprocessing_reference = block_name
        self.split = version
        db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ]
        self.meta_data = db["param"].find_one({})
        self.spatial_index = self.meta_data.pop("spatial")
        self.type_index = self.meta_data.pop("features")
        self.category_index = self.meta_data.pop("categories")
        print(f"Creating index for {db_handle}...")
        db_handle.create_index("idx")
        print("Done!")
        self.length = db_handle.estimated_document_count()
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
        ]
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
        sample.type_index[sample.type_index == -1] = len(self.type_index)
        return sample


class AirQualityDataRegression(Dataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        block_size: int,
        db_config: Path,
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
        if f"block_{block_size}" not in db.list_collection_names():
            if create_preprocessing:
                print(
                    f"No pre-processing for {block_size=}, this could take a while..."
                )
                decompose_data_regression_block(self.mongo_config, block_size)
            else:
                raise Exception(f"No preprocessing data available for {block_size=}")
        else:
            print(f"Pre-processing found for {block_size=}")
        self.preprocessing_reference = f"block_{block_size}"
        self.split = version
        db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ]
        self.meta_data = db_handle.find_one({})
        self.category_index = self.meta_data.pop("category_index")
        self.spatial_index = self.meta_data.pop("spatial_index")
        self.type_index = self.meta_data.pop("type_index")
        db_split = db_handle
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

    @staticmethod
    def adj_2_edge(adj_t: torch.Tensor):
        edge_index = adj_t.nonzero().t().contiguous()
        return edge_index

    @staticmethod
    def graph_transform(sample: dict) -> ContinuousTimeGraphSample:
        if "attention_mask" not in sample or len(sample["attention_mask"]) == 0:
            sample["time"] = torch.tensor(sample["time"], dtype=torch.float)
            sample["attention_mask"] = sample["time"].unsqueeze(-1).T < sample[
                "time"
            ].unsqueeze(-1)
        if "edge_index" in sample and len(sample["edge_index"]) == 0:
            sample.pop("edge_index")
        graph_sample = ContinuousTimeGraphSample(**sample)
        graph_sample.attention_mask = torch.logical_or(
            graph_sample.attention_mask, graph_sample.key_padding_mask.unsqueeze(0)
        )
        # adj_t = (~torch.logical_or(graph_sample.attention_mask, graph_sample.key_padding_mask.unsqueeze(1))).long()
        # graph_sample.edge_index = self.adj_2_edge(adj_t)
        graph_sample.node_features = torch.nan_to_num(graph_sample.node_features)
        return graph_sample

    def __len__(self):
        if self.subset is None:
            return self.length
        else:
            return self.subset

    def __getitem__(self, item: ContinuousTimeGraphSample):
        if not self.lazy_loaded:
            self.lazy_load_db()
        if self.shuffle:
            item = randint(0, self.length - 1)
        sample = self.graph_transform(self.db_handle.find_one({"idx": item}))
        return sample


if __name__ == "__main__":
    with open("./data/mongo_config.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
        config["data_root"] = "data/raw"
    create_interpolation_dataset(
        config, block_size=1500, sparsity=0.1, block_steps=1500, exists_ok=False
    )
