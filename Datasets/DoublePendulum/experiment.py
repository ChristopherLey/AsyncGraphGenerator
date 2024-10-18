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
from copy import deepcopy
from typing import Tuple

import matplotlib
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.model import AsynchronousGraphGenerator

matplotlib.use("Agg")


class AGGDoublePendulumExperiment(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict,
        data_params: dict,
        logging_params: dict,
    ):
        super().__init__()

        self.model_params = deepcopy(model_params)
        if "type" in self.model_params:
            self.model_params.pop("type")
        self.agg = AsynchronousGraphGenerator(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()
        self.train_RMSE = MeanSquaredError(squared=False)
        self.val_RMSE = MeanSquaredError(squared=False)
        self.calc_loss = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        y_hat, attention_list = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.calc_loss(y_hat, graph.target.features.to(self.device))
        return loss, y_hat, attention_list

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat, _ = self.forward(graph_sample)
        self.log("train_mse_loss", loss.item(), prog_bar=True)
        target = graph_sample.target.features.to(self.device)
        self.train_MAE(y_hat, target)
        self.log(
            "train_MAE",
            self.train_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_RMSE(y_hat, target)
        self.log(
            "train_RMSE",
            self.train_RMSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0])
        return loss

    def validation_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ContinuousTimeGraphSample]:
        loss, y_hat, attention_list = self.forward(graph_sample)
        self.log("val_mse_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        target = graph_sample.target.features.to(self.device)
        self.val_MAE(y_hat, target)
        self.log(
            "val_MAE",
            self.val_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_RMSE(y_hat, target)
        self.log(
            "val_RMSE",
            self.val_RMSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss.detach().to("cpu"), y_hat.detach().to("cpu"), graph_sample

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser = Adam(self.agg.parameters(), lr=self.optimiser_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimiser,
            start_factor=1.0,
            end_factor=(
                self.optimiser_params["min_lr"] / self.optimiser_params["max_lr"]
            ),
            total_iters=self.optimiser_params["total_iters"],
        )
        # fmt: off
        return [optimiser, ], [lr_scheduler, ]
        # fmt: on


class ATGDoublePendulumExperiment(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict,
        data_params: dict,
        logging_params: dict,
    ):
        super().__init__()

        self.model_params = deepcopy(model_params)
        if "type" in self.model_params:
            self.model_params.pop("type")
        self.agg = AsynchronousGraphGenerator(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()
        self.train_RMSE = MeanSquaredError(squared=False)
        self.val_RMSE = MeanSquaredError(squared=False)
        self.calc_loss = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        y_hat, attention_list = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.calc_loss(y_hat, graph.target.features.to(self.device))
        return loss, y_hat, attention_list

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat, _ = self.forward(graph_sample)
        self.log("train_mse_loss", loss.item(), prog_bar=True)
        target = graph_sample.target.features.to(self.device)
        self.train_MAE(y_hat, target)
        self.log(
            "train_MAE",
            self.train_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_RMSE(y_hat, target)
        self.log(
            "train_RMSE",
            self.train_RMSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0])
        return loss

    def validation_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, ContinuousTimeGraphSample]:
        loss, y_hat, attention_list = self.forward(graph_sample)
        self.log("val_mse_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        target = graph_sample.target.features.to(self.device)
        self.val_MAE(y_hat, target)
        self.log(
            "val_MAE",
            self.val_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_RMSE(y_hat, target)
        self.log(
            "val_RMSE",
            self.val_RMSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss.detach().to("cpu"), y_hat.detach().to("cpu"), graph_sample

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser = Adam(self.agg.parameters(), lr=self.optimiser_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimiser,
            start_factor=1.0,
            end_factor=(
                self.optimiser_params["min_lr"] / self.optimiser_params["max_lr"]
            ),
            total_iters=self.optimiser_params["total_iters"],
        )
        # fmt: off
        return [optimiser, ], [lr_scheduler, ]
        # fmt: on
