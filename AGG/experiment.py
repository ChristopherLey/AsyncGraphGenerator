from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.optim import Adam

from AGG.extended_typing import ContinuousTimeGraphSample
from AGG.model import AsynchronousGraphGenerator


class AGGExperiment(pl.LightningModule):
    def __init__(
        self,
        model_params: dict,
        optimiser_params: dict,
        data_params: dict,
        logging_params: Optional[dict] = None,
    ):
        super().__init__()

        self.model_params = model_params
        self.agg = AsynchronousGraphGenerator(**self.model_params)
        self.logging_params = logging_params
        self.optimiser_params = optimiser_params
        self.data_params = data_params
        self.batch_size = (
            self.data_params["batch_size"] if "batch_size" in data_params else None
        )

    def forward(
        self, graph: ContinuousTimeGraphSample
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, y_hat = self.agg(graph, device=self.device)
        return loss, y_hat

    def training_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> torch.Tensor:
        loss, y_hat = self.forward(graph_sample)
        self.log("train_mse_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(
        self, graph_sample: ContinuousTimeGraphSample, sample_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, y_hat = self.forward(graph_sample)
        self.log("val_mse_loss", loss.item(), prog_bar=True)
        return loss.detach().to("cpu"), y_hat.detach().to("cpu")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return Adam(self.agg.parameters(), **self.optimiser_params)
