import copy
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm import trange

from AGG.extended_typing import ContinuousTimeGraphSample

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
    "time": [],
    "edge_index": [],
    "type_index": [],
    "spatial_index": [],
    "attention_mask": [],
    "category_index": [],
}


def decompose_data(config: dict, block_size: int):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client["Beijing"]
    df: pd.DataFrame = pd.read_hdf(Path(config["data_root"]) / "pm2_5_df.h5")  # noqa
    locations = len(unique_stations)
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    block_data = db[f"block_{block_size}"]
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
                        graph["node_features"].append(line[key])
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
                                        graph_sample["node_features"].append(
                                            not_targets[key]
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
                                    graph_sample["node_features"].append(
                                        not_targets[key]
                                    )
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


class AirQualityData(Dataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        block_size: int,
        db_config: Path,
        version: str = "train",
        create_preprocessing: bool = False,
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
                decompose_data(self.mongo_config, block_size)
            else:
                raise Exception(f"No preprocessing data available for {block_size=}")
        self.preprocessing_reference = f"block_{block_size}"
        self.split = version
        db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ][self.split]
        db_handle.create_index("idx")
        self.length = db_handle.estimated_document_count()
        self.lazy_loaded = False
        self.db_handle = None

    def lazy_load_db(self):
        mongo_db_client = MongoClient(
            host=self.mongo_config["host"], port=self.mongo_config["port"]
        )
        self.db_handle = mongo_db_client[self.mongo_config["base"]][
            self.preprocessing_reference
        ][self.split]
        self.lazy_loaded = True

    @staticmethod
    def graph_transform(sample: dict):
        graph_sample = ContinuousTimeGraphSample(**sample)
        if graph_sample.attention_mask is None or len(graph_sample.attention_mask) == 0:
            graph_sample.attention_mask = graph_sample.time.unsqueeze(
                -1
            ).T < graph_sample.time.unsqueeze(-1)
        return graph_sample

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if not self.lazy_loaded:
            self.lazy_load_db()
        sample = self.graph_transform(self.db_handle.find_one({"idx": item}))
        return sample


if __name__ == "__main__":
    test_obj = AirQualityData(10, Path("./data/mongo_config.yaml"))
    print(len(test_obj))
    print(test_obj[20])
