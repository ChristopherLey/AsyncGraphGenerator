from typing import Optional
from typing import Union

import torch
from pydantic import BaseModel
from pydantic import validator
from torch import BoolTensor
from torch import FloatTensor
from torch import LongTensor


def cast_2_float_tensor(v: Union[list, FloatTensor]):
    if isinstance(v, FloatTensor):
        return v
    else:
        return torch.tensor(v, dtype=torch.float)


def cast_2_long_tensor(v: Union[list, LongTensor]):
    if isinstance(v, LongTensor):
        return v
    else:
        return torch.tensor(v, dtype=torch.long)


class TargetNode(BaseModel):
    features: FloatTensor
    type_index: LongTensor
    time: FloatTensor
    spatial_index: LongTensor

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

    class Config:
        arbitrary_types_allowed = True


class ContinuousTimeGraphSample(BaseModel):
    node_features: FloatTensor
    edge_index: Optional[LongTensor]
    attention_mask: Optional[BoolTensor]
    time: FloatTensor
    target: TargetNode
    type_index: LongTensor
    spatial_index: LongTensor
    category_index: Optional[LongTensor]

    _cast_node_features: classmethod = validator(
        "node_features", allow_reuse=True, pre=True
    )(cast_2_float_tensor)
    _cast_edge_index: classmethod = validator("edge_index", allow_reuse=True, pre=True)(
        cast_2_long_tensor
    )
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

    @validator("attention_mask", pre=True)
    def cast_attention(cls, v: Union[list, BoolTensor]):
        if isinstance(v, BoolTensor):
            return v
        else:
            return torch.tensor(v, dtype=torch.bool)

    class Config:
        arbitrary_types_allowed = True


def test_data_classes():
    target = {
        "features": [10],
        "time": [0.1],
        "type_index": [10],
        "spatial_index": [10],
        "dummy": 20,
    }
    graph = {
        "node_features": [10],
        "time": [0.1],
        "target": target,
        "type_index": [10],
        "spatial_index": [10],
        "attention_mask": [1],
        "kaboom": "kaboom",
    }
    ContinuousTimeGraphSample(**graph)
    test = ContinuousTimeGraphSample(**{**graph, "category_index": [10]})
    print(test.target.__fields__)
