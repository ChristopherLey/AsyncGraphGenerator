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
from typing import List
from typing import Optional
from typing import Union

import torch
from pydantic import BaseModel
from pydantic import validator
from pydantic.fields import ModelField
from torch import BoolTensor
from torch import FloatTensor
from torch import LongTensor


def cast_2_float_tensor(v: Union[list, FloatTensor], values: dict, field: ModelField):
    if isinstance(v, FloatTensor):
        entry = v
    else:
        entry = torch.tensor(v, dtype=torch.float)
    if "node_features" in values and entry.shape[0] != values["node_features"].shape[0]:
        raise ValueError(
            f"{field.name} shape: {entry.shape} != node_features.shape {values['node_features'].shape}"
        )
    elif "features" in values and entry.shape[0] != values["features"].shape[0]:
        raise ValueError(
            f"{field.name} shape: {entry.shape} != features.shape {values['node_features'].shape}"
        )
    return entry


def cast_2_long_tensor(v: Union[list, LongTensor], values: dict, field: ModelField):
    if isinstance(v, LongTensor):
        entry = v
    else:
        entry = torch.tensor(v, dtype=torch.long)
    if "node_features" in values and entry.shape[0] != values["node_features"].shape[0]:
        raise ValueError(
            f"{field.name} shape: {entry.shape} != node_features.shape {values['node_features'].shape}"
        )
    elif "features" in values and entry.shape[0] != values["features"].shape[0]:
        raise ValueError(
            f"{field.name} shape: {entry.shape} != features.shape {values['node_features'].shape}"
        )
    return entry


class TargetNode(BaseModel):
    features: FloatTensor
    type_index: LongTensor
    time: FloatTensor
    spatial_index: LongTensor
    category_index: Optional[LongTensor]

    _cast_features: classmethod = validator("features", allow_reuse=True, pre=True)(
        cast_2_float_tensor
    )
    _cast_type_index: classmethod = validator("type_index", allow_reuse=True, pre=True)(
        cast_2_long_tensor
    )
    _cast_time: classmethod = validator("time", allow_reuse=True, pre=True)(
        cast_2_float_tensor
    )
    _cast_spatial_index: classmethod = validator(
        "spatial_index", allow_reuse=True, pre=True
    )(cast_2_long_tensor)
    _cast_category_index: classmethod = validator(
        "category_index", allow_reuse=True, pre=True
    )(cast_2_long_tensor)

    class Config:
        arbitrary_types_allowed = True

    def unsqueeze(self, dim: int = 0):
        self.features = self.features.unsqueeze(dim)
        self.type_index = self.type_index.unsqueeze(dim)
        self.time = self.time.unsqueeze(dim)
        self.spatial_index = self.spatial_index.unsqueeze(dim)


class ContinuousTimeGraphSample(BaseModel):
    node_features: FloatTensor
    key_padding_mask: BoolTensor
    edge_index: Optional[LongTensor]
    time: FloatTensor
    attention_mask: BoolTensor
    target: TargetNode
    type_index: LongTensor
    spatial_index: LongTensor
    category_index: Optional[LongTensor]

    _cast_node_features: classmethod = validator(
        "node_features", allow_reuse=True, pre=True
    )(cast_2_float_tensor)

    _cast_time: classmethod = validator("time", allow_reuse=True, pre=True)(
        cast_2_float_tensor
    )
    _cast_type_index: classmethod = validator("type_index", allow_reuse=True, pre=True)(
        cast_2_long_tensor
    )
    _cast_spatial_index: classmethod = validator(
        "spatial_index", allow_reuse=True, pre=True
    )(cast_2_long_tensor)
    _cast_category_index: classmethod = validator(
        "category_index", allow_reuse=True, pre=True
    )(cast_2_long_tensor)

    @validator("target", pre=True)
    def create_target_node(cls, v: Union[dict, TargetNode]):
        if isinstance(v, TargetNode):
            return v
        else:
            return TargetNode(**v)

    @validator("edge_index", pre=True)
    def create_edge_index(cls, v: Union[list, LongTensor]):
        if isinstance(v, LongTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.long)
        if entry.shape[-2] != 2:
            raise ValueError()
        return entry

    @validator("key_padding_mask", pre=True)
    def create_key_padding_mask(
        cls, v: Union[list, BoolTensor], values: dict, field: ModelField
    ):
        if isinstance(v, BoolTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.bool)
        if entry.shape[0] != values["node_features"].shape[0]:
            raise ValueError(f"edge_index shape is incorrect: {entry.shape}")
        return entry

    @validator("attention_mask", pre=True)
    def cast_attention(
        cls, v: Union[list, BoolTensor], values: dict, field: ModelField
    ):
        if isinstance(v, BoolTensor):
            entry = v
        else:
            entry = torch.tensor(v, dtype=torch.bool)
        if (
            entry.shape[-1] != values["node_features"].shape[-2]
            or entry.shape[-2] != values["node_features"].shape[-2]
        ):
            raise ValueError(
                f"{field.name} shape: {entry.shape} != node_features.shape "
                f"[{values['node_features'].shape[-2]}, {values['node_features'].shape[-2]}]"
            )
        return entry

    class Config:
        arbitrary_types_allowed = True

    def unsqueeze(self, dim: int = 0):
        self.node_features = self.node_features.unsqueeze(dim)
        self.key_padding_mask = self.key_padding_mask.unsqueeze(dim)
        if self.edge_index is not None:
            self.edge_index = self.edge_index.unsqueeze(dim)
        self.time = self.time.unsqueeze(dim)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.unsqueeze(dim)
        self.target.unsqueeze(dim)
        self.type_index = self.type_index.unsqueeze(dim)
        self.spatial_index = self.spatial_index.unsqueeze(dim)
        if self.category_index is not None:
            self.category_index = self.category_index.unsqueeze(dim)


def collate_graph_samples(graph_samples: List[ContinuousTimeGraphSample]):
    for sample in graph_samples:
        sample.unsqueeze()
    batch_graph: dict = {
        "node_features": [],
        "key_padding_mask": [],
        "time": [],
        "target": {
            "features": [],
            "type_index": [],
            "time": [],
            "spatial_index": [],
        },
        "type_index": [],
        "spatial_index": [],
    }
    if graph_samples[0].attention_mask is not None:
        batch_graph["attention_mask"] = []
    if graph_samples[0].edge_index is not None:
        batch_graph["edge_index"] = []
    if graph_samples[0].category_index is not None:
        batch_graph["category_index"] = []
    if graph_samples[0].target.category_index is not None:
        batch_graph["target"]["category_index"] = []
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
