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
import io
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchmetrics import MeanAbsoluteError
from torchmetrics import MeanSquaredError
from torchmetrics import R2Score
from torchmetrics import F1Score
from torchmetrics import Accuracy
from torchmetrics import AUROC
from torchvision import transforms

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.transformer_model import AsynchronousGraphGeneratorTransformer
from AGG.graph_model import AsynchronousGraphGenerator
from AGG.utils import fig2img
from AGG.metrics import MeanRelativeError

matplotlib.use("Agg")


class AGGExperiment_PM25(pl.LightningModule):
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
        for i in range(len(attention_list) - 1):
            plt.figure(i, figsize=(1024, 1024))
            average_attention = torch.mean(attention_list[i], dim=0).to("cpu")
            sns.heatmap(average_attention, annot=True, fmt=".2f")
            img = fig2img(plt.gcf())
            self.logger.experiment.add_image(
                f"mean_attention_layer_{i}_val",
                img,
                global_step=self.global_step,
                close=True,
            )
            plt.clf()
        plt.figure("final", figsize=(1024, 1024))
        average_final_attention = torch.mean(attention_list[-1], dim=0).to("cpu")
        sns.heatmap(average_final_attention, annot=True, fmt=".2f")
        img = fig2img(plt.gcf())
        self.logger.experiment.add_image(
            "mean_final_attention_val", img, global_step=self.global_step, close=True
        )
        plt.clf()
        plt.close()
        return loss.detach().to("cpu"), y_hat.detach().to("cpu")

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


class AGGExperiment_Activity(pl.LightningModule):
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return loss.detach().to("cpu"), y_hat.detach().to("cpu")

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


class AGGExperimentICUInterpolation(pl.LightningModule):
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
        if loss.isnan().item():
            pass
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

    @staticmethod
    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        T = transforms.ToTensor()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return T(img), img

    def generate_plots(self, rmse, errors, y_scaled, type):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=200, label=f"Errors{type}\n RMSE: {rmse}")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.hist(y_scaled, bins=200, label=f"{type} data distribution")
        plt.legend()
        plt.savefig(
            Path("./lightning_logs") / self.logger.version / f"{type}.png",
            bbox_inches="tight",
        )
        plt.close()

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


class AGGExperimentICUClassification(pl.LightningModule):
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
        if data_params['pretrained_path'] is not None:
            checkpoint = torch.load(data_params['pretrained_path'])
            self.agg.load_state_dict(checkpoint["state_dict"], strict=False)
        if data_params['freeze_pretrained']:
            for name, param in self.agg.named_parameters():
                if name.startswith('head'):
                    continue
                if name.startswith('cross_attention'):
                    continue
                param.requires_grad = False
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        self.train_AUROC = AUROC(task="binary")
        self.val_AUROC = AUROC(task="binary")
        self.train_Accuracy = Accuracy(task="binary")
        self.val_Accuracy = Accuracy(task="binary")
        self.train_F1 = F1Score(task='binary')
        self.val_F1 = F1Score(task='binary')
        self.calc_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        y_hat, attention_list = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)

        loss = self.calc_loss(y_hat, graph.target.features.float().to(self.device))
        return loss, y_hat, attention_list

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat, _ = self.forward(graph_sample)
        self.log("train_BCE_loss", loss.item(), prog_bar=True)
        target = graph_sample.target.features.to(self.device)
        self.train_AUROC(y_hat, target)
        self.log(
            "train_AUROC",
            self.train_AUROC,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_F1(y_hat, target)
        self.log(
            "train_F1",
            self.train_F1,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_Accuracy(y_hat, target)
        self.log(
            "train_Accuracy",
            self.train_Accuracy,
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
        if loss.isnan().item():
            pass
        self.log("val_BCE_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        target = graph_sample.target.features.to(self.device)
        y_hat_prob = torch.sigmoid(y_hat)
        self.val_AUROC(y_hat_prob, target)
        self.log(
            "val_AUROC",
            self.val_AUROC,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_F1(y_hat_prob, target)
        self.log(
            "val_F1",
            self.val_F1,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_Accuracy(y_hat_prob, target)
        self.log(
            "val_Accuracy",
            self.val_Accuracy,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss.detach().to("cpu"), y_hat.detach().to("cpu"), graph_sample

    @staticmethod
    def fig2img(fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        T = transforms.ToTensor()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return T(img), img

    def generate_plots(self, rmse, errors, y_scaled, type):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=200, label=f"Errors{type}\n RMSE: {rmse}")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.hist(y_scaled, bins=200, label=f"{type} data distribution")
        plt.legend()
        plt.savefig(
            Path("./lightning_logs") / self.logger.version / f"{type}.png",
            bbox_inches="tight",
        )
        plt.close()

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


class AGGExperimentKDDInterpolation(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict,
        data_params: dict,
        logging_params: dict,
    ):
        super().__init__()

        self.model_params = deepcopy(model_params)
        type = self.model_params.pop("type")
        if type == "Transformer":
            self.agg = AsynchronousGraphGeneratorTransformer(**self.model_params)
        else:
            self.agg = AsynchronousGraphGenerator(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        if logging_params["scaling"] is not None:
            self.scaler = logging_params["scaling"]
        else:
            self.scaler = None
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()
        self.train_MAE_PM25 = MeanAbsoluteError()
        self.val_MAE_PM25 = MeanAbsoluteError()
        self.train_RMSE = MeanSquaredError(squared=False)
        self.val_RMSE = MeanSquaredError(squared=False)
        self.train_PM25_RMSE = MeanSquaredError(squared=False)
        self.val_PM25_RMSE = MeanSquaredError(squared=False)
        self.train_PM25_MSE = MeanSquaredError(squared=True)
        self.val_PM25_MSE = MeanSquaredError(squared=True)
        self.train_PM25_MRE = MeanRelativeError()
        self.val_PM25_MRE = MeanRelativeError()
        self.calc_loss = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        y_hat, attention_list = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.calc_loss(y_hat, graph.target.features.to(self.device))
        return loss, y_hat, attention_list

    def rescale_normal_data(self, data: torch.Tensor, key: str = 'PM2.5'):
        if self.scaler is None:
            return data
        scaling = self.scaler[key]
        return data*scaling['std'] + scaling['mean']

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
        pm25_mask = graph_sample.target.type_index == 0
        if pm25_mask.any():
            y_hat_PM25 = self.rescale_normal_data(y_hat[pm25_mask], key='PM2.5')
            target_PM25 = self.rescale_normal_data(target[pm25_mask], key='PM2.5')
            self.train_PM25_RMSE(y_hat_PM25, target_PM25)
            self.log(
                "train_PM25_RMSE",
                self.train_PM25_RMSE,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.train_MAE_PM25(y_hat_PM25, target_PM25)
            self.log(
                "train_MAE_PM25",
                self.train_MAE_PM25,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.train_PM25_MSE(y_hat_PM25, target_PM25)
            self.log(
                "train_PM25_MSE",
                self.train_PM25_MSE,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.train_PM25_MRE(y_hat_PM25, target_PM25)
            self.log(
                "train_PM25_MRE",
                self.train_PM25_MRE,
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
        pm25_mask = graph_sample.target.type_index == 0
        if pm25_mask.any():
            y_hat_PM25 = self.rescale_normal_data(y_hat[pm25_mask], key='PM2.5')
            target_PM25 = self.rescale_normal_data(target[pm25_mask], key='PM2.5')
            self.val_PM25_RMSE(y_hat_PM25, target_PM25)
            self.log(
                "val_PM25_RMSE",
                self.val_PM25_RMSE,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.val_MAE_PM25(y_hat_PM25, target_PM25)
            self.log(
                "val_MAE_PM25",
                self.val_MAE_PM25,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.val_PM25_MSE(y_hat_PM25, target_PM25)
            self.log(
                "val_PM25_MSE",
                self.val_PM25_MSE,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.val_PM25_MRE(y_hat_PM25, target_PM25)
            self.log(
                "val_PM25_MRE",
                self.val_PM25_MRE,
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


class AGGExperimentAQIInterpolation(pl.LightningModule):
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
        self.agg = AsynchronousGraphGenerator(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )
        if logging_params["scaling"] is not None:
            self.scaler = logging_params["scaling"]
        else:
            self.scaler = None

        # Metrics
        self.train_MAE = MeanAbsoluteError()
        self.val_MAE = MeanAbsoluteError()
        self.train_RMSE = MeanSquaredError(squared=False)
        self.val_RMSE = MeanSquaredError(squared=False)
        self.train_MSE = MeanSquaredError(squared=True)
        self.val_MSE = MeanSquaredError(squared=True)
        self.train_MRE = MeanRelativeError()
        self.val_MRE = MeanRelativeError()
        # the same but with normalised
        self.train_MAE_norm = MeanAbsoluteError()
        self.val_MAE_norm = MeanAbsoluteError()
        self.train_RMSE_norm = MeanSquaredError(squared=False)
        self.val_RMSE_norm = MeanSquaredError(squared=False)
        self.train_MSE_norm = MeanSquaredError(squared=True)
        self.val_MSE_norm = MeanSquaredError(squared=True)
        self.train_MRE_norm = MeanRelativeError()
        self.val_MRE_norm = MeanRelativeError()

        self.calc_loss = nn.MSELoss()

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        y_hat, attention_list = self.agg(graph, device=self.device)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.unsqueeze(0)
        loss = self.calc_loss(y_hat, graph.target.features.to(self.device))
        return loss, y_hat, attention_list

    def rescale_normal_data(self, data: torch.Tensor):
        if self.scaler is None:
            return data
        scaling = self.scaler
        return data*scaling['std'] + scaling['mean']

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat, _ = self.forward(graph_sample)
        self.log("train_mse_loss", loss.item(), prog_bar=True)
        target = graph_sample.target.features.to(self.device)

        # Unnormalised metrics
        y_hat_scaled = self.rescale_normal_data(y_hat)
        target_scaled = self.rescale_normal_data(target)
        self.train_MRE(y_hat_scaled, target_scaled)
        self.log(
            "train_MRE",
            self.train_MRE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_RMSE(y_hat_scaled, target_scaled)
        self.log(
            "train_RMSE",
            self.train_RMSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_MAE(y_hat_scaled, target_scaled)
        self.log(
            "train_MAE",
            self.train_MAE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_MSE(y_hat_scaled, target_scaled)
        self.log(
            "train_MSE",
            self.train_MSE,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        # Normalised metrics
        self.train_MRE_norm(y_hat, target)
        self.log(
            "train_MRE_norm",
            self.train_MRE_norm,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_RMSE_norm(y_hat, target)
        self.log(
            "train_RMSE_norm",
            self.train_RMSE_norm,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.train_MAE_norm(y_hat, target)
        self.log(
            "train_MAE_norm",
            self.train_MAE_norm,
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

        # Unnormalised metrics
        y_hat_scaled = self.rescale_normal_data(y_hat)
        target_scaled = self.rescale_normal_data(target)
        self.val_MRE(y_hat_scaled, target_scaled)
        self.log(
            "val_MRE",
            self.val_MRE,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_RMSE(y_hat_scaled, target_scaled)
        self.log(
            "val_RMSE",
            self.val_RMSE,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_MAE(y_hat_scaled, target_scaled)
        self.log(
            "val_MAE",
            self.train_MAE,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_MSE(y_hat_scaled, target_scaled)
        self.log(
            "val_MSE",
            self.val_MSE,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        # Normalised metrics
        self.val_MRE_norm(y_hat, target)
        self.log(
            "val_MRE_norm",
            self.train_MRE_norm,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_RMSE_norm(y_hat, target)
        self.log(
            "val_RMSE_norm",
            self.train_RMSE_norm,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.val_MAE_norm(y_hat, target)
        self.log(
            "val_MAE_norm",
            self.train_MAE_norm,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        return loss.detach().to("cpu"), y_hat.detach().to("cpu"), graph_sample

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimiser = Adam(self.agg.parameters(), lr=self.optimiser_params["lr"])
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer=optimiser,
            start_factor=self.optimiser_params["start_factor"],
            end_factor=self.optimiser_params["end_factor"],
            total_iters=self.optimiser_params["total_iters"],
        )
        # fmt: off
        return [optimiser, ], [lr_scheduler, ]
        # fmt: on