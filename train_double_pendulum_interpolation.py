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
import argparse
import os
import sys
from datetime import datetime
from math import floor
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from AGG.extended_typing import collate_graph_samples
from Datasets.DoublePendulum.datareader import DoublePendulumDataset
from Datasets.DoublePendulum.experiment import AGGDoublePendulumExperiment


def main():
    parser = argparse.ArgumentParser(description="Generic runner for AGG")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="double_pendulum_config.yaml",
    )
    parser.add_argument(
        "--resume-ckpt",
        "-ckpt",
        dest="ckpt",
        metavar="FILE",
        help="path to checkpoint",
        default=None,
    )
    args = parser.parse_args()
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        args.filename = (
            Path(f"{ckpt_path.parts[0]}/{ckpt_path.parts[1]}") / "config.yaml"
        )
    else:
        ckpt_path = None
    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        print("Debugging Mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        config["data_params"]["num_workers"] = 0
        config["data_params"]["persistent_workers"] = False
    else:
        config["data_params"]["persistent_workers"] = True
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    if (
        "subset" in config["data_params"]
        and config["data_params"]["subset"] is not None
    ):
        assert config["data_params"]["subset"] < 1.0
        train_reader = DoublePendulumDataset(
            block_size=config["data_params"]["block_size"],
            sparsity=config["data_params"]["sparsity"],
            db_config=Path(config["data_params"]["db_config"]),
            dynamics_params=config['data_params']["dynamics_params"],
            generation_params=config['data_params']["generation_params"],
            create_preprocessing=True,
            version="train",
        )
        train_length = len(train_reader)
        subset = floor(train_length * config["data_params"]["subset"])
        print(f"Training with a subset of {subset}/{train_length}")
        print(f'Total train dataset length: {train_length}')
        shuffle = True
    else:
        subset = None
        shuffle = False
    train_reader = DoublePendulumDataset(
        block_size=config["data_params"]["block_size"],
        sparsity=config["data_params"]["sparsity"],
        db_config=Path(config["data_params"]["db_config"]),
        dynamics_params=config['data_params']["dynamics_params"],
        generation_params=config['data_params']["generation_params"],
        version="train",
        subset=subset,
        shuffle=shuffle,
    )
    train_dataloader = DataLoader(
        train_reader,
        shuffle=False,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
        persistent_workers=config["data_params"]["persistent_workers"],
    )
    val_reader = DoublePendulumDataset(
        block_size=config["data_params"]["block_size"],
        sparsity=config["data_params"]["sparsity"],
        db_config=Path(config["data_params"]["db_config"]),
        dynamics_params=config['data_params']["dynamics_params"],
        generation_params=config['data_params']["generation_params"],
        version="test",
    )
    val_dataloader = DataLoader(
        val_reader,
        shuffle=False,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
        persistent_workers=config["data_params"]["persistent_workers"],
    )

    config["model_params"]["num_node_types"] = len(train_reader.type_index)
    config["model_params"]["num_spatial_components"] = len(train_reader.spatial_index)
    config["model_params"]["num_categories"] = len(train_reader.category_index)

    model = AGGDoublePendulumExperiment(
        model_params=config["model_params"],
        optimiser_params=config["optimiser_params"],
        data_params=config["data_params"],
        logging_params=config["logging_params"],
    )
    mse_callback = ModelCheckpoint(
        monitor="val_mse_loss",
        save_top_k=4,
        mode="min",
        filename="model-{epoch:02d}-{val_mse_loss:.6f}",
    )
    mae_callback = ModelCheckpoint(
        monitor="val_RMSE_epoch",
        save_top_k=4,
        mode="min",
        filename="model-{epoch:02d}-{val_RMSE_epoch:.6f}",
    )

    callbacks = [mse_callback, mae_callback]

    version_path = (
        f"AGG-dble_pend_{int(config['data_params']['sparsity'] * 100)}%_"
        f"inter-{datetime.now().strftime('%d-%m_%H:%M:%S')}"
    )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=".",
        version=version_path,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=1000,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )

    pprint(config)
    for key, value in config.items():
        trainer.logger.experiment.add_text(key, str(value), global_step=0)

    log_path = Path(tb_logger.log_dir)
    with open(log_path / "config.yaml", "w") as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    sys.exit(main())
