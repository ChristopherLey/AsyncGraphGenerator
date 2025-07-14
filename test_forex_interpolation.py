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
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from AGG.experiments import AGGExperimentFXInterpolation
from AGG.extended_typing import collate_graph_samples
from Datasets.Foreign_Exchange_Rates.datareader import ForexInterpolationDataset


def main():
    parser = argparse.ArgumentParser(description="Generic runner for AGG")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="fx_config.yaml",
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
    config = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        ckpt_config_path = Path(f"{Path(args.ckpt).parent.parent}") / "config.yaml"
        args.filename = Path(f"{ckpt_path.parent.parent}") / "config.yaml"
        with open(ckpt_config_path, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        ckpt_path = None
    if config is None:
        with open(args.filename, "r") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        print("Debugging Mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        config["data_params"]["num_workers"] = 0
        persistent_workers = False
    else:
        persistent_workers = True
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    test_reader = ForexInterpolationDataset(
        block_size=config["data_params"]["block_size"],
        sparsity=config["data_params"]["sparsity"],
        db_config=Path(config["data_params"]["db_config"]),
        version="test",
    )
    test_dataloader = DataLoader(
        test_reader,
        shuffle=False,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
        persistent_workers=persistent_workers,
    )

    config["model_params"]["num_node_types"] = len(test_reader.type_index)
    config["model_params"]["num_spatial_components"] = len(test_reader.spatial_index)
    config["model_params"]["num_categories"] = len(test_reader.category_index)

    model = AGGExperimentFXInterpolation(
        model_params=config["model_params"],
        optimiser_params=config["optimiser_params"],
        data_params=config["data_params"],
        logging_params=config["logging_params"],
    )

    version_path = (
        f"AGG-fx_{config['model_params']['type']}_{int(config['data_params']['sparsity'] * 100)}%_"
        f"steps_{config['data_params']['block_steps']}_"
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
        max_epochs=1000,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
    )

    pprint(config)
    for key, value in config.items():
        trainer.logger.experiment.add_text(key, str(value), global_step=0)

    trainer.test(model, test_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    sys.exit(main())
