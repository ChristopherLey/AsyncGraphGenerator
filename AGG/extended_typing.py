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
import torch
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import ValidationInfo
from torch import BoolTensor
from torch import FloatTensor
from torch import LongTensor


def cast_2_float_tensor(v: list | FloatTensor, info: ValidationInfo, **kwargs):
    if isinstance(v, FloatTensor):
        entry = v
    elif isinstance(v, float):
        entry = torch.tensor([v], dtype=torch.float)
    else:
        entry = torch.tensor(v, dtype=torch.float)
    if (
        "node_features" in info.data
        and entry.shape[0] != info.data["node_features"].shape[0]
    ):
        raise ValueError(
            f"{info.field_name} shape: {entry.shape} != node_features.shape {info.data['node_features'].shape}"
        )
    elif "features" in info.data and entry.shape[0] != info.data["features"].shape[0]:
        raise ValueError(
            f"{info.field_name} shape: {entry.shape} != features.shape {info.data['node_features'].shape}"
        )
    return torch.nan_to_num(entry)


def cast_2_long_tensor(v: list | LongTensor, info: ValidationInfo, **kwargs):
    if isinstance(v, LongTensor):
        entry = v
    elif isinstance(v, int):
        entry = torch.tensor([v], dtype=torch.long)
    else:
        entry = torch.tensor(v, dtype=torch.long)
    if (
        "node_features" in info.data
        and entry.shape[0] != info.data["node_features"].shape[0]
    ):
        raise ValueError(
            f"{info.field_name} shape: {entry.shape} != node_features.shape {info.data['node_features'].shape}"
        )
    elif "features" in info.data and entry.shape[0] != info.data["features"].shape[0]:
        raise ValueError(
            f"{info.field_name} shape: {entry.shape} != features.shape {info.data['node_features'].shape}"
        )
    return entry


class TargetNode(BaseModel):
    features: LongTensor | FloatTensor
    type_index: LongTensor | None = None
    time: FloatTensor
    spatial_index: LongTensor | None = None
    category_index: LongTensor | FloatTensor | None = None

    _cast_type_index: classmethod = field_validator("type_index", mode="before")(
        cast_2_long_tensor
    )
    _cast_time: classmethod = field_validator("time", mode="before")(
        cast_2_float_tensor
    )
    _cast_spatial_index: classmethod = field_validator("spatial_index", mode="before")(
        cast_2_long_tensor
    )

    @field_validator("features", mode="before")
    @classmethod
    def create_features(cls, v: dict | LongTensor | FloatTensor, info: ValidationInfo):
        if isinstance(v, LongTensor):
            entry = v
        elif isinstance(v, FloatTensor):
            entry = v
        elif isinstance(v, float):
            entry = torch.tensor([v], dtype=torch.float)
        else:
            if isinstance(v[0], int):
                entry = torch.tensor(v, dtype=torch.long)
            else:
                entry = torch.tensor(v, dtype=torch.float)
                entry = torch.nan_to_num(entry)
        return entry

    @field_validator("category_index", mode="before")
    @classmethod
    def create_category_index(
        cls, v: dict | LongTensor | FloatTensor, info: ValidationInfo
    ):
        if isinstance(v, LongTensor):
            entry = v
        elif isinstance(v, FloatTensor):
            entry = v
        else:
            if isinstance(v[0], int):
                entry = torch.tensor(v, dtype=torch.long)
            else:
                entry = torch.tensor(v, dtype=torch.float)
                entry = torch.nan_to_num(entry)
        return entry

    class Config:
        arbitrary_types_allowed = True

    def unsqueeze(self, dim: int = 0):
        self.features = self.features.unsqueeze(dim)
        if self.type_index is not None:
            self.type_index = self.type_index.unsqueeze(dim)
        self.time = self.time.unsqueeze(dim)
        if self.spatial_index is not None:
            self.spatial_index = self.spatial_index.unsqueeze(dim)


class ContinuousTimeGraphSample(BaseModel):
    node_features: FloatTensor
    key_padding_mask: BoolTensor | None = None
    edge_index: LongTensor | None = None
    time: FloatTensor
    attention_mask: BoolTensor
    target: TargetNode
    type_index: LongTensor | None = None
    spatial_index: LongTensor | None = None
    category_index: LongTensor | FloatTensor | None = None

    _cast_node_features: classmethod = field_validator("node_features", mode="before")(
        cast_2_float_tensor
    )

    _cast_time: classmethod = field_validator("time", mode="before")(
        cast_2_float_tensor
    )
    _cast_type_index: classmethod = field_validator("type_index", mode="before")(
        cast_2_long_tensor
    )
    _cast_spatial_index: classmethod = field_validator("spatial_index", mode="before")(
        cast_2_long_tensor
    )

    @field_validator("category_index", mode="before")
    @classmethod
    def create_category_index(
        cls, v: dict | LongTensor | FloatTensor, info: ValidationInfo
    ):
        if isinstance(v, LongTensor):
            entry = v
        elif isinstance(v, FloatTensor):
            entry = v
        else:
            if isinstance(v[0], int):
                entry = torch.tensor(v, dtype=torch.long)
            else:
                entry = torch.tensor(v, dtype=torch.float)
                entry = torch.nan_to_num(entry)
        return entry

    @field_validator("target", mode="before")
    @classmethod
    def create_target_node(cls, v: dict | TargetNode):
        if isinstance(v, TargetNode):
            return v
        else:
            return TargetNode(**v)

    @field_validator("edge_index", mode="before")
    @classmethod
    def create_edge_index(cls, v: list | LongTensor):
        if isinstance(v, LongTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.long)
        if entry.shape[-2] != 2:
            raise ValueError()
        return entry

    @field_validator("key_padding_mask", mode="before")
    @classmethod
    def create_key_padding_mask(cls, v: list | BoolTensor, info: ValidationInfo):
        if isinstance(v, BoolTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.bool)
        if entry.shape[0] != info.data["node_features"].shape[0]:
            raise ValueError(f"key_padding_mask shape is incorrect: {entry.shape}")
        return entry

    @field_validator("attention_mask", mode="before")
    @classmethod
    def cast_attention(cls, v: list | BoolTensor, info: ValidationInfo):
        if isinstance(v, BoolTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.bool)
        if (
            entry.shape[-1] != info.data["time"].shape[-1]
            or entry.shape[-2] != info.data["time"].shape[-1]
        ):
            raise ValueError(
                f"{info.field_name} shape: {entry.shape} != "
                f"[{info.data['time'].shape[-1]}, {info.data['time'].shape[-1]}]"
            )
        return entry

    class Config:
        arbitrary_types_allowed = True

    def unsqueeze(self, dim: int = 0):
        self.node_features = self.node_features.unsqueeze(dim)
        self.time = self.time.unsqueeze(dim)
        if self.key_padding_mask is not None:
            self.key_padding_mask = self.key_padding_mask.unsqueeze(dim)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.unsqueeze(dim)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.unsqueeze(dim)
        if self.type_index is not None:
            self.type_index = self.type_index.unsqueeze(dim)
        if self.spatial_index is not None:
            self.spatial_index = self.spatial_index.unsqueeze(dim)
        if self.category_index is not None:
            self.category_index = self.category_index.unsqueeze(dim)
        self.target.unsqueeze(dim)


def collate_graph_samples(graph_samples: list[ContinuousTimeGraphSample]):
    for sample in graph_samples:
        if len(sample.node_features.shape) == 1:
            sample.unsqueeze()
    batch_graph: dict = {
        "node_features": [],
        "time": [],
        "target": {
            "features": [],
            "time": [],
        },
    }
    if graph_samples[0].key_padding_mask is not None:
        batch_graph["key_padding_mask"] = []
    if graph_samples[0].type_index is not None:
        batch_graph["type_index"] = []
    if graph_samples[0].attention_mask is not None:
        batch_graph["attention_mask"] = []
    if graph_samples[0].edge_index is not None:
        batch_graph["edge_index"] = []
    if graph_samples[0].category_index is not None:
        batch_graph["category_index"] = []
    if graph_samples[0].target.category_index is not None:
        batch_graph["target"]["category_index"] = []
    if graph_samples[0].target.type_index is not None:
        batch_graph["target"]["type_index"] = []
    if graph_samples[0].target.spatial_index is not None:
        batch_graph["target"]["spatial_index"] = []
    if graph_samples[0].spatial_index is not None:
        batch_graph["spatial_index"] = []
    for sample in graph_samples:
        for key in batch_graph.keys():
            if key != "target":
                batch_graph[key].append(sample.__getattribute__(key))
            else:
                for target_key in batch_graph[key].keys():
                    batch_graph[key][target_key].append(
                        sample.target.__getattribute__(target_key)
                    )
    for key in batch_graph.keys():
        if key != "target":
            batch_graph[key] = torch.cat(batch_graph[key], dim=0)
        else:
            for target_key in batch_graph[key].keys():
                batch_graph[key][target_key] = torch.cat(
                    batch_graph[key][target_key], dim=0
                )
    return ContinuousTimeGraphSample(**batch_graph)


def test_data_classes():
    target = {
        "features": [10],
        "time": [0.1],
        "type_index": [10],
        "spatial_index": [10],
        "dummy": 20,
    }
    graph = {
        "node_features": [10, 20, 30],
        "attention_mask": [
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ],
        "key_padding_mask": [False, True, False],
        "time": [0.9, 0.5, 0.1],
        "target": target,
        "type_index": [10, 10, 10],
        "spatial_index": [10, 10, 10],
        "kaboom": "kaboom",
    }
    test = ContinuousTimeGraphSample(**graph)
    test2 = ContinuousTimeGraphSample(**graph)
    collate_graph_samples([test, test2])
    ContinuousTimeGraphSample(**{**graph, "category_index": [10, 10, 10]})
