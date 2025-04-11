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
from .data.data_masks import test_masks
from .data.data_masks import training_masks
from .data.data_masks import validation_masks
from dateutil.relativedelta import relativedelta
from pymongo import MongoClient
from tqdm import trange

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.graph_dataset import GraphDataset
from Datasets.data_tools import random_index

time_scale = 60.0
features = [
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$",
    "EURO AREA - EURO/US$",
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$",
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$",
    "BRAZIL - REAL/US$",
    "CANADA - CANADIAN DOLLAR/US$",
    "CHINA - YUAN/US$",
    "HONG KONG - HONG KONG DOLLAR/US$",
    "INDIA - INDIAN RUPEE/US$",
    "KOREA - WON/US$",
    "MEXICO - MEXICAN PESO/US$",
    "SOUTH AFRICA - RAND/US$",
    "SINGAPORE - SINGAPORE DOLLAR/US$",
    "DENMARK - DANISH KRONE/US$",
    "JAPAN - YEN/US$",
    "MALAYSIA - RINGGIT/US$",
    "NORWAY - NORWEGIAN KRONE/US$",
    "SWEDEN - KRONA/US$",
    "SRI LANKA - SRI LANKAN RUPEE/US$",
    "SWITZERLAND - FRANC/US$",
    "TAIWAN - NEW TAIWAN DOLLAR/US$",
    "THAILAND - BAHT/US$",
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

#%%
def normalise(feature, scaling):
    return (feature - scaling["mean"]) / scaling["std"]


def decompose_raw_forex_data(config: dict, replace: bool = False):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    collections = db.list_collection_names()
    replace_params = False
    lines = None
    header = None
    if "param" in collections:
        if db["param"].find_one() is not None:
            if replace:
                db["param"].drop()
                replace_params = True
        else:
            replace_params = True
    else:
        replace_params = True
    if replace_params:
        file_path = Path(config["data_root"])
        assert file_path.exists()
        with open(file_path, "r") as f:
            lines = f.readlines()
        header = lines[0][:-1].split(",")
        scale_dictionary = {}
        for line in lines[1:]:
            split_line = line[:-1].split(",")
            for category, value in zip(header[2:], split_line[2:]):
                try:
                    f_value = float(value)
                    if category not in scale_dictionary:
                        scale_dictionary[category] = [
                            f_value,
                        ]
                    else:
                        scale_dictionary[category].append(f_value)
                except ValueError:
                    continue
        param = {}
        for category in header[2:]:
            param[category] = {
                "mean": np.mean(scale_dictionary[category]),
                "std": np.std(scale_dictionary[category]),
            }
        db["param"].insert_one(param)
    else:
        param = db["param"].find_one()
    replace_raw = False
    if "raw" in collections:
        if db["raw"].find_one() is not None:
            if replace:
                db["raw"].drop()
                replace_raw = True
        else:
            replace_raw = True
    else:
        replace_raw = True
    if replace_raw:
        if lines is None:
            file_path = Path(config["data_root"])
            assert file_path.exists()
            with open(file_path, "r") as f:
                lines = f.readlines()
            header = lines[0][:-1].split(",")
        raw_graph = []
        idx = 0
        for line in lines[1:]:
            split_line = line[:-1].split(",")
            time_stamp = datetime.strptime(split_line[1], "%Y-%m-%d")
            for category, value in zip(header[2:], split_line[2:]):
                try:
                    f_value = float(value)
                    graph_proto = {
                        "idx": idx,
                        "time": time_stamp,
                        "node_features": normalise(f_value, param[category]),
                        "type_index": features.index(category),
                    }
                    raw_graph.append(graph_proto)
                    idx += 1
                except ValueError:
                    continue
        db["raw"].insert_many(raw_graph)

def create_forex_data_block(
    raw: list[dict],
    input_index: list,
    target_index: list,
    sample_count: int,
):
    write_data = []
    max_time = raw[input_index[-1]]["time"]
    graph = copy.deepcopy(graph_template)
    for k in range(len(input_index)):
        raw_entry = raw[input_index[k]]
        graph["node_features"].append(raw_entry["node_features"])
        graph["time"].append(
            (max_time - raw_entry["time"]).days / time_scale
        )
        graph["type_index"].append(raw_entry["type_index"])
    graph["key_padding_mask"] = (np.zeros_like(input_index) != 0).tolist()
    removed_index_array = np.array(target_index)
    mask = (removed_index_array > min(input_index)) & (
        removed_index_array < max(input_index)
    )
    graph_target = removed_index_array[mask]
    for i in range(graph_target.shape[0]):
        k = graph_target[i]
        graph_sample = copy.deepcopy(graph)
        target = copy.deepcopy(target_template)
        target["features"] = [
            raw[k]["node_features"],
        ]
        target["type_index"] = [
            raw[k]["type_index"],
        ]
        target["time"] = [
            (max_time - raw[k]["time"]).days / time_scale,
        ]
        graph_sample["target"] = target
        graph_sample["idx"] = sample_count
        sample_count += 1
        write_data.append(graph_sample)
    return write_data, sample_count

def create_forex_dataset(config: dict, block_size: int, sparsity=0.3, block_steps: int = 22,
                         replace_dataset: bool=False,
                         replace_raw: bool=False):
    assert 0 < sparsity < 1.0
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    decompose_raw_forex_data(config, replace=replace_raw)
    db_raw = db["raw"]
    block_name = f"block_{block_size:02d}_{100 * sparsity}%"
    existing_collections = db.list_collection_names(
        filter={"name": {"$regex": block_name}}
    )
    if replace_dataset or len(existing_collections) == 0:
        for entry in existing_collections:
            db.drop_collection(entry)
    else:
        if len(existing_collections) > 0:
            return
    block_db = db[block_name]
    test_block = block_db["test"]
    train_block = block_db["train"]
    validation_block = block_db["validation"]
    # filter the training sets
    train_sample_count = 0
    min_len = np.inf
    for start_date in training_masks:
        cursor = db_raw.find({
            "time": {
                "$gte": start_date,
                "$lt": start_date + relativedelta(months=2)}
        })
        samples = list(cursor)
        removed, remainder = random_index(len(samples), sparsity)
        min_len = min(min_len, len(remainder))
        for n in trange(0, len(remainder) - block_size, block_steps):
            write_data, train_sample_count = create_forex_data_block(
                samples,
                remainder[n: (n + block_size)],
                removed,
                train_sample_count,
            )
            if len(write_data) > 0:
                train_block.insert_many(write_data)
    # filter out the testing sets
    test_sample_count = 0
    for start_date in test_masks:
        cursor = db_raw.find({
            "time": {
                "$gte": start_date,
                "$lt": start_date + relativedelta(months=2)}
        })
        samples = list(cursor)
        removed, remainder = random_index(len(samples), sparsity)
        min_len = min(min_len, len(remainder))
        for n in trange(0, len(remainder) - block_size, block_steps):
            write_data, test_sample_count = create_forex_data_block(
                samples,
                remainder[n: (n + block_size)],
                removed,
                test_sample_count,
            )
            if len(write_data) > 0:
                test_block.insert_many(write_data)
    # filter out the validation sets
    validation_sample_count = 0
    for start_date in validation_masks:
        cursor = db_raw.find({
            "time": {
                "$gte": start_date,
                "$lt": start_date + relativedelta(months=2)}
        })
        samples = list(cursor)
        removed, remainder = random_index(len(samples), sparsity)
        min_len = min(min_len, len(remainder))
        for n in trange(0, len(remainder) - block_size, block_steps):
            write_data, validation_sample_count = create_forex_data_block(
                samples,
                remainder[n: (n + block_size)],
                removed,
                validation_sample_count,
            )
            if len(write_data) > 0:
                validation_block.insert_many(write_data)
    print(min_len)



class ForexInterpolationDataset(GraphDataset):
    valid_versions = ["train", "test", "validation"]
    type_index = features
    spatial_index = []
    category_index = []

    def __init__(
        self,
        block_size: int,
        sparsity: float,
        db_config: Path,
        block_steps: int = 22,
        version: str = "train",
        subset: Optional[int] = None,
        shuffle: bool = False,
        replace_dataset: bool = False,
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
        self.lazy_loaded = False
        self.db_handle = None
        block_name = f"block_{block_size:02d}_{100 * sparsity}%"
        if block_name not in db.list_collection_names() or replace_dataset:
            create_forex_dataset(self.mongo_config, block_size, sparsity, block_steps, replace_dataset=replace_dataset)
        else:
            print(f"Pre-processing found for {block_name=}")

        self.preprocessing_reference = block_name
        self.split = version
        db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ][self.split]
        self.meta_data = db["param"].find_one({})
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
    import os
    current_path = os.getcwd()
    test_obj = ForexInterpolationDataset(
        block_size=450,
        sparsity=0.3,
        db_config= Path(current_path) / Path('Foreign_Exchange_Rates/data/mongo_config.yaml'),
        version="test",
    )
    print(len(test_obj))  # 1895393
    assert isinstance(len(test_obj), int)
    sample: ContinuousTimeGraphSample = test_obj[1]
    assert not sample.node_features.isnan().any()
    assert not sample.target.features.isnan().any()