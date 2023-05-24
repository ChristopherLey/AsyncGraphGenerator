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

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.transformer_model import AsynchronousGraphGeneratorTransformer


class AGGExperiment(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict,
        data_params: dict,
        logging_params: dict,
    ):
        super().__init__()

        self.model_params = deepcopy(model_params)
        self.model_params.pop("type")
        self.agg = AsynchronousGraphGeneratorTransformer(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()
        self.train_R2 = R2Score()
        self.val_R2 = R2Score()
        self.train_pm25_MAE = MeanAbsoluteError()
        self.val_pm25_MAE = MeanAbsoluteError()
        self.train_pm25_R2 = R2Score()
        self.val_pm25_R2 = R2Score()
        self.train_pm25_MSE = MeanSquaredError()
        self.val_pm25_MSE = MeanSquaredError()
        self.calc_loss = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.calc_loss(y_hat, graph.target.features.to(self.device))
        return loss, y_hat

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat = self.forward(graph_sample)
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
        is_pm25 = graph_sample.target.type_index == 0
        pm25_min = self.logging_params["scaler"]["min"][0]
        pm25_max = self.logging_params["scaler"]["max"][0]
        pm25_target = self.inverse_minmax_scale(target[is_pm25], pm25_min, pm25_max)
        pm25_predict = self.inverse_minmax_scale(y_hat[is_pm25], pm25_min, pm25_max)
        self.train_pm25_MAE(pm25_predict, pm25_target)
        self.log(
            "train_pm25_MAE",
            self.train_pm25_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_pm25_MSE(pm25_predict, pm25_target)
        self.log(
            "train_pm25_MSE",
            self.train_pm25_MSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        scheduler = self.lr_schedulers()
        self.log("lr", scheduler.get_last_lr()[0])
        # self.train_pm25_R2(pm25_predict, pm25_target)
        # self.log("train_pm25_R2", self.train_pm25_R2, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    @staticmethod
    def inverse_minmax_scale(x: torch.Tensor, x_min: float, x_max: float):
        return x * (x_max - x_min) + x_min

    def validation_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, y_hat = self.forward(graph_sample)
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
        is_pm25 = graph_sample.target.type_index == 0
        pm25_min = self.logging_params["scaler"]["min"][0]
        pm25_max = self.logging_params["scaler"]["max"][0]
        pm25_target = self.inverse_minmax_scale(target[is_pm25], pm25_min, pm25_max)
        pm25_predict = self.inverse_minmax_scale(y_hat[is_pm25], pm25_min, pm25_max)
        self.val_pm25_MAE(pm25_predict, pm25_target)
        self.log(
            "val_pm25_MAE",
            self.val_pm25_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_pm25_MSE(pm25_predict, pm25_target)
        self.log(
            "val_pm25_MSE",
            self.val_pm25_MSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        # self.val_pm25_R2(pm25_predict, pm25_target)
        # self.log("val_pm25_R2", self.val_pm25_R2, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss.detach().to("cpu"), y_hat.detach().to("cpu")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser = Adam(self.agg.parameters(), lr=self.optimiser_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimiser,
            start_factor=(
                self.optimiser_params["min_lr"] / self.optimiser_params["max_lr"]
            ),
            end_factor=1.0,
            total_iters=self.optimiser_params["total_iters"],
        )
        # fmt: off
        return [optimiser, ], [lr_scheduler, ]
