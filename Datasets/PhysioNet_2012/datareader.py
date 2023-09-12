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
import math
from pathlib import Path
from random import randint
from typing import Dict
from typing import Optional

import numpy as np
import yaml
from pymongo import MongoClient
from tqdm import tqdm
from tqdm import trange

from AGG.graph_dataset import GraphDataset

params: Dict[str, dict] = {
    "Age": {"range": [15, 90.0]},
    "Gender": {"categories": 2, "range": [0, 1], "minLimit": None, "maxLimit": None},
    "Height": {"range": [121.9, 462.3], "minLimit": 100, "maxLimit": 300},
    "ICUType": {
        "categories": 4,
        "range": [1, 2, 3, 4],
        "minLimit": None,
        "maxLimit": None,
    },
    "Weight": {
        "range": [-1.0, 472.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 80.0,
        "iqr": 76.3,
    },
    "GCS": {"range": [3.0, 15.0], "minLimit": None, "maxLimit": None},
    "HR": {
        "range": [0.0, 300.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 86.0,
        "iqr": 60.0,
    },
    "NIDiasABP": {
        "range": [-1.0, 211.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 57.0,
        "iqr": 51.0,
    },
    "NIMAP": {
        "range": [0.0, 228.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 75.33,
        "iqr": 50.33,
    },
    "NISysABP": {
        "range": [0.0, 300.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 117.0,
        "iqr": 74.0,
    },
    "Temp": {
        "range": [18.6, 42.2],
        "minLimit": 0,
        "maxLimit": None,
        "median": 37.1,
        "iqr": 2.799999999999997,
    },
    "Urine": {
        "range": [0.0, 1695.0],
        "minLimit": None,
        "maxLimit": 1700,
        "median": 70.0,
        "iqr": 368.0,
    },
    "SaO2": {
        "range": [0.0, 100.0],
        "minLimit": 0,
        "maxLimit": None,
        "median": 97.0,
        "iqr": 7.0,
    },
    "FiO2": {"range": [0.21, 1.0], "minLimit": None, "maxLimit": None},
    "DiasABP": {
        "range": [-1.0, 283.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 58.0,
        "iqr": 41.0,
    },
    "MAP": {
        "range": [0.0, 300.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 77.0,
        "iqr": 50.0,
    },
    "SysABP": {"range": [0.0, 295.0], "minLimit": None, "maxLimit": None},
    "pH": {
        "range": [0, 14],
        "minLimit": 0,
        "maxLimit": 14,
        "median": 7.38,
        "iqr": 0.25,
    },
    "PaCO2": {"range": [0.0, 100.0], "minLimit": None, "maxLimit": None},
    "PaO2": {"range": [0.0, 500.0], "minLimit": None, "maxLimit": None},
    "MechVent": {"range": [0, 2], "minLimit": None, "maxLimit": None},
    "BUN": {
        "range": [0.0, 209.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 20.0,
        "iqr": 67.0,
    },
    "Creatinine": {
        "range": [0.1, 22.1],
        "minLimit": None,
        "maxLimit": None,
        "median": 1.0,
        "iqr": 3.9000000000000004,
    },
    "Glucose": {
        "range": [8.0, 599.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 127.0,
        "iqr": 166.0,
    },
    "HCO3": {"range": [5.0, 52.0], "minLimit": None, "maxLimit": None},
    "HCT": {
        "range": [5.0, 61.8],
        "minLimit": None,
        "maxLimit": None,
        "median": 30.2,
        "iqr": 16.400000000000002,
    },
    "Mg": {
        "range": [0.0, 4.9],
        "minLimit": None,
        "maxLimit": None,
        "median": 2.0,
        "iqr": 1.3000000000000003,
    },
    "Platelets": {
        "range": [5.0, 2292.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 172.0,
        "iqr": 323.0,
    },
    "K": {
        "range": [1.5, 13.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 4.1,
        "iqr": 2.0999999999999996,
    },
    "Na": {
        "range": [98.0, 180.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 139.0,
        "iqr": 16.0,
    },
    "WBC": {
        "range": [0.0, 528.0],
        "minLimit": None,
        "maxLimit": 529,
        "median": 11.5,
        "iqr": 20.299999999999997,
    },
    "ALP": {
        "range": [8.0, 4695.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 82.0,
        "iqr": 263.0,
    },
    "ALT": {
        "range": [1.0, 16920.0],
        "minLimit": None,
        "maxLimit": None,
        "median": 42.0,
        "iqr": 1893.0999999999985,
    },
    "AST": {"range": [4.0, 36400.0], "minLimit": None, "maxLimit": None},
    "Bilirubin": {"range": [0.0, 82.8], "minLimit": None, "maxLimit": None},
    "RespRate": {"range": [0.0, 100.0], "minLimit": None, "maxLimit": None},
    "Lactate": {"range": [0.0, 31.0], "minLimit": None, "maxLimit": None},
    "Albumin": {"range": [1.0, 5.3], "minLimit": None, "maxLimit": None},
    "TroponinT": {"range": [0.01, 29.91], "minLimit": None, "maxLimit": None},
    "TroponinI": {"range": [0.1, 49.6], "minLimit": None, "maxLimit": None},
    "Cholesterol": {"range": [28.0, 362.0], "minLimit": None, "maxLimit": None},
}
types = [
    "Weight",
    "GCS",
    "HR",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Temp",
    "Urine",
    "SaO2",
    "FiO2",
    "DiasABP",
    "MAP",
    "SysABP",
    "pH",
    "PaCO2",
    "PaO2",
    "MechVent",
    "BUN",
    "Creatinine",
    "Glucose",
    "HCO3",
    "HCT",
    "Mg",
    "Platelets",
    "K",
    "Na",
    "WBC",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "RespRate",
    "Lactate",
    "Albumin",
    "TroponinT",
    "TroponinI",
    "Cholesterol",
    "<PAD>",
]
Ignore = ["RecordID"]
spatial = ["ICUType"]
category = ["Age", "Height", "Gender"]

max_time = 48 * 60.0
unique_ICUType = [0, 1, 2, 3, 4]

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
normalisation = {
    "Weight": {"mean": 83.04956714219259, "std": 24.8646751214909},
    "GCS": {"mean": 11.405530375125679, "std": 3.9989943567956403},
    "HR": {"mean": 87.45116782170582, "std": 18.535902208845524},
    "NIDiasABP": {"mean": 58.16823843899262, "std": 15.655314651724245},
    "NIMAP": {"mean": 77.05823725627891, "std": 15.777383532288622},
    "NISysABP": {"mean": 119.08149276595168, "std": 23.578671920810283},
    "Temp": {"mean": 37.01894019372885, "std": 1.5990062705380956},
    "Urine": {"mean": 120.86300737177226, "std": 178.27610061650165},
    "SaO2": {"mean": 96.66067114205896, "std": 3.647787649147071},
    "FiO2": {"mean": 0.5461345691077416, "std": 0.19175182938620708},
    "DiasABP": {"mean": 59.26860387699593, "std": 13.365668053240013},
    "MAP": {"mean": 79.71379881286389, "std": 17.08875774109639},
    "SysABP": {"mean": 118.87069613711482, "std": 25.163413188253845},
    "pH": {"mean": 7.420648967551623, "std": 4.858785157363492},
    "PaCO2": {"mean": 40.477165160468076, "std": 9.202362380263905},
    "PaO2": {"mean": 150.44212354819976, "std": 88.81701401375705},
    "BUN": {"mean": 27.176413941789434, "std": 22.59802301831056},
    "Creatinine": {"mean": 1.4730657317305857, "std": 1.5496184576367908},
    "Glucose": {"mean": 140.9512895438953, "std": 65.386418215852},
    "HCO3": {"mean": 23.151272513703994, "std": 4.726606954127707},
    "HCT": {"mean": 30.677117025340632, "std": 5.010691399135171},
    "Mg": {"mean": 2.02356729006234, "std": 0.5168551816751211},
    "Platelets": {"mean": 189.71520281359906, "std": 107.01311606231748},
    "K": {"mean": 4.129157430471445, "std": 0.6818755378759372},
    "Na": {"mean": 139.10642712427077, "std": 5.218685838038974},
    "WBC": {"mean": 13.258554795928859, "std": 64.18740414345862},
    "ALP": {"mean": 120.09268086007839, "std": 175.12346489251092},
    "ALT": {"mean": 362.8910710607621, "std": 1132.7878992858052},
    "AST": {"mean": 505.18049835255357, "std": 1649.0809292739586},
    "Bilirubin": {"mean": 2.8639645151422455, "std": 5.76925913039896},
    "RespRate": {"mean": 19.631111917067127, "std": 5.511857026613568},
    "Lactate": {"mean": 3.0004955250841614, "std": 2.6024155368409416},
    "Albumin": {"mean": 2.8903021106359503, "std": 0.6524478697445716},
    "TroponinT": {"mean": 1.1736666133674447, "std": 2.7853700937925097},
    "TroponinI": {"mean": 8.210382978723406, "std": 10.973640918036587},
    "Cholesterol": {"mean": 155.678391959799, "std": 44.46017689352526},
}


def min_max(value: float, scale_range: list):
    return (value - scale_range[0]) / (scale_range[1] - scale_range[0])


def parse_minutes(time_str: str) -> float:
    hour, minute = time_str.split(":")
    return float(hour) * 60.0 + float(minute)


def normalise_physionet(config: dict, exist_ok: bool = True):
    root_path = Path(config["data_root"])
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    set_a = root_path / "set-a"
    set_b = root_path / "set-b"
    set_c = root_path / "set-c"
    raw_datasets = [set_a, set_b, set_c]
    scale_db = db["normalised_data"]
    db_len = scale_db.estimated_document_count()
    if not exist_ok and db_len > 0:
        scale_db.drop()
    print("Normalising physionet_data")
    for dir_loc in raw_datasets:
        for entry in dir_loc.iterdir():
            with open(entry, "r") as f:
                data_str_list: list[str] = f.readlines()
            _, name, record_id_str = data_str_list[1][:-1].split(",")
            record_id = int(record_id_str)
            graph_entry = copy.deepcopy(graph_template)
            assert name == "RecordID"
            graph_entry["idx"] = record_id
            category_entry = []
            spatial = None
            for i in range(2, len(data_str_list)):
                t_str, target, feature_str = data_str_list[i][:-1].split(",")
                if len(target) == 0:
                    print(f"empty entry {record_id=}, {i}: {data_str_list[i]}")
                    continue
                if target == "ICUType":
                    spatial = int(feature_str)
                    continue
                elif target == "Age":
                    scaler = params[target]["range"]
                    category_entry.append(min_max(float(feature_str), scaler))
                    continue
                elif target == "Gender":
                    category_entry.append(float(feature_str))
                    continue
                elif target == "Height":
                    height = float(feature_str)
                    scaler = params[target]["range"]
                    if height > params[target]["maxLimit"]:
                        category_entry.append(1.0)
                    elif height < params[target]["minLimit"]:
                        if height >= 10:
                            category_entry.append(min_max(height * 10.0, scaler))
                        elif height <= 0:
                            category_entry.append(0)
                        else:
                            category_entry.append(min_max(height * 100.0, scaler))
                    else:
                        category_entry.append(min_max(height, scaler))
                    continue
                feature = float(feature_str)
                if math.isnan(feature):
                    continue
                tau = (48 * 60.0 - parse_minutes(t_str)) / (48 * 60.0)
                if target in normalisation:
                    feature = (feature - normalisation[target]["mean"]) / normalisation[
                        target
                    ]["std"]
                else:
                    feature = feature
                graph_entry["node_features"].append(feature)
                graph_entry["time"].append(tau)
                graph_entry["type_index"].append(types.index(target))
            graph_entry["category_index"] = [
                category_entry,
            ] * len(graph_entry["node_features"])
            graph_entry["spatial_index"] = [
                spatial,
            ] * len(graph_entry["node_features"])
            if len(graph_entry["node_features"]) <= 10:
                print(graph_entry["idx"], len(graph_entry["node_features"]))
            else:
                scale_db.insert_one(graph_entry)


def scale_physionet(config: dict, exist_ok: bool = True):
    root_path = Path(config["data_root"])
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    set_a = root_path / "set-a"
    set_b = root_path / "set-b"
    set_c = root_path / "set-c"
    raw_datasets = [set_a, set_b, set_c]
    scale_db = db["scaled_data"]
    db_len = scale_db.estimated_document_count()
    if not exist_ok and db_len > 0:
        scale_db.drop()
    for dir_loc in raw_datasets:
        for entry in dir_loc.iterdir():
            with open(entry, "r") as f:
                data_str_list: list[str] = f.readlines()
            _, name, record_id_str = data_str_list[1][:-1].split(",")
            record_id = int(record_id_str)
            graph_entry = copy.deepcopy(graph_template)
            assert name == "RecordID"
            graph_entry["idx"] = record_id
            category_entry = []
            spatial = None
            for i in range(2, len(data_str_list)):
                t_str, target, feature_str = data_str_list[i][:-1].split(",")
                if len(target) == 0:
                    print(f"empty entry {record_id=}, {i}: {data_str_list[i]}")
                    continue
                if target == "ICUType":
                    spatial = int(feature_str)
                    continue
                elif target == "Age":
                    scaler = params[target]["range"]
                    category_entry.append(min_max(float(feature_str), scaler))
                    continue
                elif target == "Gender":
                    category_entry.append(float(feature_str))
                    continue
                elif target == "Height":
                    height = float(feature_str)
                    scaler = params[target]["range"]
                    if height > params[target]["maxLimit"]:
                        category_entry.append(1.0)
                    elif height < params[target]["minLimit"]:
                        if height >= 10:
                            category_entry.append(min_max(height * 10.0, scaler))
                        elif height <= 0:
                            category_entry.append(0)
                        else:
                            category_entry.append(min_max(height * 100.0, scaler))
                    else:
                        category_entry.append(min_max(height, scaler))
                    continue
                feature = float(feature_str)
                if (
                    params[target]["minLimit"] is not None
                    and feature < params[target]["minLimit"]
                ):
                    continue
                if (
                    params[target]["maxLimit"] is not None
                    and feature > params[target]["maxLimit"]
                ):
                    continue
                tau = (48 * 60.0 - parse_minutes(t_str)) / (48 * 60.0)
                feature = min_max(feature, params[target]["range"])
                graph_entry["node_features"].append(feature)
                graph_entry["time"].append(tau)
                graph_entry["type_index"].append(types.index(target))
            graph_entry["category_index"] = [
                category_entry,
            ] * len(graph_entry["node_features"])
            graph_entry["spatial_index"] = [
                spatial,
            ] * len(graph_entry["node_features"])
            if len(graph_entry["node_features"]) <= 10:
                print(graph_entry["idx"], len(graph_entry["node_features"]))
            else:
                scale_db.insert_one(graph_entry)


def decompose_physionet_data_interpolation(
    config: dict,
    block_size: int,
    exists_ok: bool = False,
    sparsity: float = 0.5,
    skip_leq_block: bool = True,
    normal: bool = False,
    block_steps: int = 1,
):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    if normal:
        scale_db = db["normalised_data"]
    else:
        scale_db = db["scaled_data"]
    if scale_db.estimated_document_count() == 0:
        if normal:
            normalise_physionet(config)
        else:
            scale_physionet(config)
    assert scale_db.estimated_document_count() > 0
    data_cursor = scale_db.find({})
    if skip_leq_block:
        block_name = f"block_{block_size:02d}_{100 * sparsity}%_skip"
    else:
        block_name = f"block_{block_size:02d}_{100 * sparsity}%"
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
        "type_index": types,
        "scaling": params,
        "spatial_index": unique_ICUType,
        "spatial": spatial,
        "category": category,
        "time": max_time,
        "normal": normal,
    }
    if normal:
        meta["normalisation"] = normalisation
    block_db.insert_one(meta)
    indexes = {}
    for entry in data_cursor:
        removed = np.random.choice(
            len(entry["time"]),
            size=int(np.floor(len(entry["time"]) * sparsity)),
            replace=False,
        )
        removed.sort()
        remainder_bool = np.array([i not in removed for i in range(len(entry["time"]))])
        remainder = np.arange(len(entry["time"]))[remainder_bool]
        test_index = np.random.choice(
            removed.shape[0], size=removed.shape[0] // 5, replace=False
        )
        test_index.sort()
        test = removed[test_index]
        train_index = np.array([i not in test_index for i in range(removed.shape[0])])
        train = removed[train_index]
        indexes[entry["idx"]] = {
            "removed": removed,
            "remainder": remainder,
            "train": train,
            "test": test,
        }
    test_sample_count = 0
    train_sample_count = 0
    for idx, index in tqdm(indexes.items()):
        data = scale_db.find_one({"idx": {"$eq": idx}})
        if index["remainder"].shape[0] < block_size:
            if skip_leq_block:
                continue
            input_index = index["remainder"]
            padding_size = block_size - index["remainder"].shape[0]
            graph = copy.deepcopy(graph_template)
            graph["node_features"] = np.array(data["node_features"])[input_index]
            graph["node_features"] = np.pad(
                graph["node_features"], (0, padding_size), "constant"
            )
            graph["node_features"] = graph["node_features"].tolist()
            graph["key_padding_mask"] = (
                np.pad(np.ones_like(index["remainder"]), (0, padding_size), "constant")
                != 1
            ).tolist()
            time = np.array(data["time"])[input_index]
            relative_time = time.min()
            graph["time"] = np.pad(
                (time - relative_time), (0, padding_size), "constant", constant_values=0
            ).tolist()
            graph["type_index"] = np.pad(
                np.array(data["type_index"])[input_index],
                (0, padding_size),
                "constant",
                constant_values=(len(types) - 1),
            ).tolist()
            graph["spatial_index"] = np.pad(
                np.array(data["spatial_index"])[input_index], (0, padding_size)
            ).tolist()
            graph["category_index"] = np.pad(
                np.array(data["category_index"])[input_index],
                ((0, padding_size), (0, 0)),
            ).tolist()
            assert (
                len(graph["category_index"]) == block_size
                and len(graph["category_index"][0]) == 3
            ), f"{graph['category_index'].shape=}"
            write_data = []
            test_set = index["test"]
            for i in range(test_set.shape[0]):
                graph_sample = copy.deepcopy(graph)
                target = copy.deepcopy(target_template)
                target["features"] = [
                    data["node_features"][test_set[i]],
                ]
                target["type_index"] = [
                    data["type_index"][test_set[i]],
                ]
                target["spatial_index"] = [
                    data["spatial_index"][test_set[i]],
                ]
                target["category_index"] = [
                    data["category_index"][test_set[i]],
                ]
                target["time"] = [
                    data["time"][test_set[i]] - relative_time,
                ]
                graph_sample["target"] = target
                graph_sample["idx"] = test_sample_count
                test_sample_count += 1
                write_data.append(graph_sample)
            if len(write_data) > 0:
                test_block.insert_many(write_data)
            train_set = index["train"]
            write_data = []
            for i in range(train_set.shape[0]):
                graph_sample = copy.deepcopy(graph)
                target = copy.deepcopy(target_template)
                target["features"] = [
                    data["node_features"][train_set[i]],
                ]
                target["type_index"] = [
                    data["type_index"][train_set[i]],
                ]
                target["spatial_index"] = [
                    data["spatial_index"][train_set[i]],
                ]
                target["category_index"] = [
                    data["category_index"][train_set[i]],
                ]
                target["time"] = [
                    data["time"][train_set[i]] - relative_time,
                ]
                graph_sample["target"] = target
                graph_sample["idx"] = train_sample_count
                train_sample_count += 1
                write_data.append(graph_sample)
            if len(write_data) > 0:
                train_block.insert_many(write_data)
        else:
            for n in trange(0, index["remainder"].shape[0] - block_size, block_steps):
                input_index = index["remainder"][n : (n + block_size)]
                graph = copy.deepcopy(graph_template)
                graph["node_features"] = np.array(data["node_features"])[
                    input_index
                ].tolist()
                graph["key_padding_mask"] = (
                    np.zeros_like(np.array(data["type_index"])[input_index]) != 0
                ).tolist()
                time = np.array(data["time"])[input_index]
                relative_time = time.min()
                graph["time"] = (time - relative_time).tolist()
                graph["type_index"] = np.array(data["type_index"])[input_index].tolist()
                graph["spatial_index"] = np.array(data["spatial_index"])[
                    input_index
                ].tolist()
                graph["category_index"] = np.array(data["category_index"])[
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
                    if test_set[i] == types.index("MechVent"):
                        continue
                    graph_sample = copy.deepcopy(graph)
                    target = copy.deepcopy(target_template)
                    target["features"] = [
                        data["node_features"][test_set[i]],
                    ]
                    target["type_index"] = [
                        data["type_index"][test_set[i]],
                    ]
                    target["spatial_index"] = [
                        data["spatial_index"][test_set[i]],
                    ]
                    target["category_index"] = [
                        data["category_index"][test_set[i]],
                    ]
                    target["time"] = [
                        data["time"][test_set[i]] - relative_time,
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
                    if train_set[i] == types.index("MechVent"):
                        continue
                    graph_sample = copy.deepcopy(graph)
                    target = copy.deepcopy(target_template)
                    target["features"] = [
                        data["node_features"][train_set[i]],
                    ]
                    target["type_index"] = [
                        data["type_index"][train_set[i]],
                    ]
                    target["spatial_index"] = [
                        data["spatial_index"][train_set[i]],
                    ]
                    target["category_index"] = [
                        data["category_index"][train_set[i]],
                    ]
                    target["time"] = [
                        data["time"][train_set[i]] - relative_time,
                    ]
                    graph_sample["target"] = target
                    graph_sample["idx"] = train_sample_count
                    train_sample_count += 1
                    write_data.append(graph_sample)
                if len(write_data) > 0:
                    train_block.insert_many(write_data)
    return train_block, test_block


def construct_outcome_physionet(config: dict, exists_ok: bool = True):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    scale_db = db["normalised_data"]
    if scale_db.estimated_document_count() == 0:
        normalise_physionet(config)
    assert scale_db.estimated_document_count() > 0
    outcome_db = db["outcomes"]
    outcome_alive = outcome_db['alive']
    outcome_dead = outcome_db['dead']
    if outcome_db['alive'].estimated_document_count() > 0:
        if exists_ok:
            return
        else:
            outcome_db.drop()
            outcome_db['alive'].drop()
            outcome_db['dead'].drop()
    root_path = Path(config["data_root"])
    set_a = root_path / "Outcomes" / "Outcomes-a.txt"
    set_b = root_path / "Outcomes" / "Outcomes-b.txt"
    set_c = root_path / "Outcomes" / "Outcomes-c.txt"
    raw_datasets = [set_a, set_b, set_c]
    for dir_loc in raw_datasets:
        with open(dir_loc, "r") as f:
            data_str_list: list[str] = f.readlines()
        for i in range(1, len(data_str_list)):
            record_id_str, SAPS_I, SOFA, Length_of_stay, Survival, In_hospital_death = data_str_list[i][:-1].split(",")
            record_id = int(record_id_str)
            SAPS_I = int(SAPS_I)
            SOFA = int(SOFA)
            Length_of_stay = int(Length_of_stay)
            Survival = int(Survival)
            In_hospital_death = int(In_hospital_death)
            outcome_db = {
                "idx": record_id,
                "SAPS_I": SAPS_I,
                "SOFA": SOFA,
                "Length_of_stay": Length_of_stay,
                "Survival": Survival,
                "In_hospital_death": In_hospital_death,
            }
            if In_hospital_death == 0:
                outcome_alive.insert_one(outcome_db)
            else:
                outcome_dead.insert_one(outcome_db)


def create_graph(data: dict, outcome: dict,  longest_node: int, mask_ratio: float = 0.0):
    node_features = np.array(data["node_features"])
    if mask_ratio > 0.0:
        index = np.arange(node_features.shape[0])
        np.random.shuffle(index)
        mask = index[int(np.floor(node_features.shape[0] * mask_ratio)):]
        mask.sort()
    else:
        mask = np.arange(node_features.shape[0])
    padding_size = longest_node - mask.shape[0]
    graph = copy.deepcopy(graph_template)
    graph["node_features"] = node_features[mask]
    graph["node_features"] = np.pad(
        graph["node_features"], (0, padding_size), "constant"
    )
    graph["node_features"] = graph["node_features"].tolist()
    graph["key_padding_mask"] = (
            np.pad(np.ones(mask.shape[0]), (0, padding_size), "constant") != 1).tolist()
    time = np.array(data["time"])
    relative_time = time.min()
    time = time[mask]
    graph["time"] = np.pad(
        (time - relative_time), (0, padding_size), "constant", constant_values=0
    ).tolist()
    graph["type_index"] = np.pad(
        np.array(data["type_index"])[mask],
        (0, padding_size),
        "constant",
        constant_values=(len(types) - 1),
    ).tolist()
    graph["spatial_index"] = np.pad(
        np.array(data["spatial_index"])[mask], (0, padding_size)
    ).tolist()
    graph["category_index"] = np.pad(
        np.array(data["category_index"])[mask, :],
        ((0, padding_size), (0, 0)), ).tolist()
    assert (
            len(graph["category_index"]) == longest_node
            and len(graph["category_index"][0]) == 3
    ), f"{graph['category_index'].shape=}"
    target = copy.deepcopy(target_template)
    target.pop("type_index")

    target["features"] = [
        outcome["In_hospital_death"],
    ]
    target["spatial_index"] = [
        data["spatial_index"][0],
    ]
    target["category_index"] = [
        data["category_index"][0],
    ]
    target["time"] = [
        0
    ]
    graph["target"] = target
    return graph


def decompose_physionet_data_k_fold(
    config: dict,
    k: int,
    exists_ok: bool = False,
    longest_node: int = 1497,
    masked_ratio: float = 0.1,
    class_bias_correction: int = 3,
    resample_rate: int = 10,
):
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    scale_db = db["normalised_data"]
    if scale_db.estimated_document_count() == 0:
        normalise_physionet(config)
    assert scale_db.estimated_document_count() > 0
    outcome_db = db["outcomes"]
    if outcome_db['alive'].estimated_document_count() == 0 or not exists_ok:
        construct_outcome_physionet(config, exists_ok)
    # create block db
    if masked_ratio > 0.0:
        block_name = f"classification_{k}-fold-masked_{masked_ratio * 100:2g}%"
    else:
        block_name = f"classification_{k}-fold-masked-balanced"
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
    # create meta data
    meta = {
        "type_index": types,
        "scaling": params,
        "spatial_index": unique_ICUType,
        "spatial": spatial,
        "category": category,
        "time": max_time,
        "normalisation": normalisation,
        "longest_node": longest_node,
        'k-fold': k,
    }
    dead_count = outcome_db['dead'].estimated_document_count()
    alive_count = outcome_db['alive'].estimated_document_count()
    test_count_alive = int(np.floor(alive_count//k))
    test_count_dead = int(np.floor(dead_count//k))
    dead_idx = np.arange(dead_count)
    np.random.shuffle(dead_idx)
    alive_idx = np.arange(alive_count)
    np.random.shuffle(alive_idx)
    k_indexes = []
    for i in range(k):
        if i == k-1:
            test_dead_idx = dead_idx[i*test_count_dead:]
            test_alive_idx = alive_idx[i*test_count_alive:]
            k_indexes.append((test_dead_idx, test_alive_idx))
        else:
            test_dead_idx = dead_idx[i * test_count_dead:(i + 1) * test_count_dead]
            test_alive_idx = alive_idx[i * test_count_alive:(i + 1) * test_count_alive]
            k_indexes.append((test_dead_idx, test_alive_idx))
    block_db.insert_one(meta)
    for k_fold in range(k):
        k_db_test = test_block[k_fold]
        k_db_train = train_block[k_fold]
        train_index_count = 0
        test_index_count = 0
        test_dead_idx, test_alive_idx = k_indexes[k_fold]

        alive_cursor = outcome_db['alive'].find({})
        for n, outcome in tqdm(enumerate(alive_cursor)):
            data = scale_db.find_one({"idx": {"$eq": outcome["idx"]}})
            if data is None:
                continue
            if n in test_alive_idx:
                graph = create_graph(data, outcome, longest_node)
                graph["idx"] = test_index_count
                test_index_count += 1
                k_db_test.insert_one(graph)
            else:
                for i in range(resample_rate):
                    graph = create_graph(data, outcome, longest_node, masked_ratio)
                    graph_copy = copy.deepcopy(graph)
                    graph_copy["idx"] = train_index_count
                    train_index_count += 1
                    k_db_train.insert_one(graph_copy)
        dead_cursor = outcome_db['dead'].find({})
        for m, outcome in tqdm(enumerate(dead_cursor)):
            data = scale_db.find_one({"idx": {"$eq": outcome["idx"]}})
            if data is None:
                continue
            if m in test_dead_idx:
                graph = create_graph(data, outcome, longest_node)
                graph["idx"] = test_index_count
                test_index_count += 1
                k_db_test.insert_one(graph)
            else:
                for i in range(resample_rate * class_bias_correction):
                    graph = create_graph(data, outcome, longest_node, masked_ratio)
                    graph_copy = copy.deepcopy(graph)
                    graph_copy["idx"] = train_index_count
                    train_index_count += 1
                    k_db_train.insert_one(graph_copy)


def decompose_physionet_data_classification(
    config: dict,
    exists_ok: bool = False,
    longest_node: int = 1497,
    test_fraction: float = 0.2,
    masked_ratio: float = 0.1,
    class_bias_correction: int = 3,
    resample_rate: int = 10,
):
    # access mongo db and get data
    mongo_db_client = MongoClient(host=config["host"], port=config["port"])
    db = mongo_db_client[config["base"]]
    scale_db = db["normalised_data"]
    if scale_db.estimated_document_count() == 0:
        normalise_physionet(config)
    assert scale_db.estimated_document_count() > 0
    outcome_db = db["outcomes"]
    if outcome_db['alive'].estimated_document_count() == 0 or not exists_ok:
        construct_outcome_physionet(config, exists_ok)
    # create block db
    if masked_ratio > 0.0:
        block_name = f"classification_{test_fraction*100:2g}%-masked_{masked_ratio*100:2g}%"
    else:
        block_name = f"classification_{test_fraction*100:2g}%-balanced"
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
    # create meta data
    meta = {
        "type_index": types,
        "scaling": params,
        "spatial_index": unique_ICUType,
        "spatial": spatial,
        "category": category,
        "time": max_time,
        "normalisation": normalisation,
        "longest_node": longest_node,
        'test_fraction': test_fraction,
    }
    dead_count = outcome_db['dead'].estimated_document_count()
    alive_count = outcome_db['alive'].estimated_document_count()
    test_count_alive = int(np.floor(alive_count * test_fraction))
    test_count_dead = int(np.floor(dead_count * test_fraction))
    dead_idx = np.arange(dead_count)
    np.random.shuffle(dead_idx)
    alive_idx = np.arange(alive_count)
    np.random.shuffle(alive_idx)
    test_dead_idx = dead_idx[:test_count_dead]
    test_alive_idx = alive_idx[:test_count_alive]

    block_db.insert_one(meta)
    train_index_count = 0
    test_index_count = 0
    alive_cursor = outcome_db['alive'].find({})
    for n, outcome in tqdm(enumerate(alive_cursor)):
        data = scale_db.find_one({"idx": {"$eq": outcome["idx"]}})
        if data is None:
            continue
        if n in test_alive_idx:
            graph = create_graph(data, outcome, longest_node)
            graph["idx"] = test_index_count
            test_index_count += 1
            test_block.insert_one(graph)
        else:
            for i in range(resample_rate):
                graph = create_graph(data, outcome, longest_node, masked_ratio)
                graph_copy = copy.deepcopy(graph)
                graph_copy["idx"] = train_index_count
                train_index_count += 1
                train_block.insert_one(graph_copy)

    dead_cursor = outcome_db['dead'].find({})
    for m, outcome in tqdm(enumerate(dead_cursor)):
        data = scale_db.find_one({"idx": {"$eq": outcome["idx"]}})
        if data is None:
            continue
        if m in test_dead_idx:
            graph = create_graph(data, outcome, longest_node)
            graph["idx"] = test_index_count
            test_index_count += 1
            test_block.insert_one(graph)
        else:
            for i in range(resample_rate * class_bias_correction):
                graph = create_graph(data, outcome, longest_node, masked_ratio)
                graph_copy = copy.deepcopy(graph)
                graph_copy["idx"] = train_index_count
                train_index_count += 1
                train_block.insert_one(graph_copy)


class ICUData(GraphDataset):
    valid_versions = ["train", "test"]

    def __init__(
        self,
        db_config: Path,
        block_size: int = 100,
        sparsity: float = 0.0,
        version: str = "train",
        create_preprocessing: bool = True,
        shuffle: bool = False,
        subset: Optional[int] = None,
        skip_leq_block: bool = False,
        normal: bool = True,
        force_preprocessing: bool = False,
        classification: bool = False,
        k_fold: int = 5,
        test_fraction: float = 0.2,
        k_fold_index: Optional[int] = None,
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
        if classification:
            if k_fold is not None:
                if sparsity > 0.0:
                    block_name = f"classification_{k_fold}-fold-masked_{sparsity * 100:2g}%"
                else:
                    block_name = f"classification_{k_fold}-fold-masked-balanced"
            else:
                if sparsity > 0.0:
                    block_name = f"classification_{test_fraction * 100:2g}%-masked_{sparsity * 100:2g}%"
                else:
                    block_name = f"classification_{test_fraction*100:2g}%-balanced"
        else:
            if skip_leq_block:
                block_name = f"block_{block_size:02d}_{100 * sparsity}%_skip"
            else:
                block_name = f"block_{block_size:02d}_{100 * sparsity}%"
        if block_name not in db.list_collection_names() or force_preprocessing:
            if create_preprocessing:
                print(
                    f"No pre-processing for {block_name=}, this could take a while..."
                )
                if classification:
                    if k_fold is not None:
                        decompose_physionet_data_k_fold(
                            self.mongo_config,
                            k=k_fold,
                            exists_ok=False,
                            masked_ratio=sparsity,
                        )
                    else:
                        decompose_physionet_data_classification(
                            self.mongo_config,
                            exists_ok=False,
                            test_fraction=test_fraction,
                            masked_ratio=sparsity,
                        )
                else:
                    decompose_physionet_data_interpolation(
                        self.mongo_config,
                        block_size=block_size,
                        sparsity=sparsity,
                        exists_ok=False,
                        normal=normal,
                        skip_leq_block=skip_leq_block,
                        block_steps=5,
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
        self.spatial_index = self.meta_data.pop("spatial_index")
        self.type_index = self.meta_data.pop("type_index")
        if self.type_index[-1] != "<PAD>":
            self.type_index.append("<PAD>")
        db_split = db_handle[self.split]
        if k_fold is not None:
            assert k_fold_index is not None
            db_split = db_split[k_fold_index]
            self.k = k_fold_index
        else:
            self.k = None
        print(f"Creating index for {db_split}...")
        db_split.create_index("idx")
        print("Done!")
        self.length = db_split.estimated_document_count()
        self.lazy_loaded = False
        self.db_handle = None
        self.shuffle = shuffle
        self.normalisation = normalisation
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
        if self.k is not None:
            mongo_db_client = MongoClient(
                host=self.mongo_config["host"], port=self.mongo_config["port"]
            )
            self.db_handle = mongo_db_client[self.mongo_config["base"]][
                self.preprocessing_reference
            ][self.split][self.k]
        else:
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
        sample.type_index[sample.type_index == -1] = len(self.type_index)
        return sample


if __name__ == "__main__":
    test_obj = ICUData(
        db_config=Path("./data/mongo_config.yaml"),
        version="train",
        classification=True,
        force_preprocessing=False,
        sparsity=0.1,
        k_fold=5,
        k_fold_index=0
    )
    print(len(test_obj))  # 11701077
    assert isinstance(len(test_obj), int)
    print(test_obj[0])

