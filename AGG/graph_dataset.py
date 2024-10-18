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
from torch.utils.data import Dataset

from AGG.extended_typing import ContinuousTimeGraphSample


class GraphDataset(Dataset):
    @staticmethod
    def adj_2_edge(adj_t: torch.Tensor):
        edge_index = adj_t.nonzero().t().contiguous()
        return edge_index

    @staticmethod
    def graph_transform(sample: dict):
        if "attention_mask" not in sample or len(sample["attention_mask"]) == 0:
            sample["time"] = torch.tensor(sample["time"], dtype=torch.float)
            sample["attention_mask"] = sample["time"].unsqueeze(-1).T < sample[
                "time"
            ].unsqueeze(-1)
        if "edge_index" in sample and len(sample["edge_index"]) == 0:
            sample.pop("edge_index")
        graph_sample = ContinuousTimeGraphSample(**sample)
        # adj_t = (~torch.logical_or(graph_sample.attention_mask, graph_sample.key_padding_mask.unsqueeze(1))).long()
        # graph_sample.edge_index = self.adj_2_edge(adj_t)
        graph_sample.node_features = torch.nan_to_num(graph_sample.node_features)
        return graph_sample
