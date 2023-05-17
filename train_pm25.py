import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from AGG.experiment import AGGExperiment
from AGG.extended_typing import collate_graph_samples
from Datasets.Beijing.datareader import AirQualityData


def main():
    parser = argparse.ArgumentParser(description="Generic runner for AGG")
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="pm25_config.yaml",
    )
    args = parser.parse_args()
    with open(args.filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if hasattr(sys, "gettrace") and sys.gettrace() is not None:
        print("Debugging Mode")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        config["num_workers"] = 0

    train_reader = AirQualityData(
        block_size=config["data_params"]["block_size"],
        db_config=Path(config["data_params"]["db_config"]),
        version="train",
    )
    train_dataloader = DataLoader(
        train_reader,
        shuffle=True,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
    )
    val_reader = AirQualityData(
        block_size=config["data_params"]["block_size"],
        db_config=Path(config["data_params"]["db_config"]),
        version="test",
    )
    val_dataloader = DataLoader(
        val_reader,
        shuffle=True,
        batch_size=config["data_params"]["batch_size"],
        drop_last=False,
        num_workers=config["data_params"]["num_workers"],
        collate_fn=collate_graph_samples,
    )

    config["model_params"]["num_node_types"] = len(train_reader.type_index)
    config["model_params"]["num_spatial_components"] = len(train_reader.spatial_index)
    config["model_params"]["num_categories"] = len(train_reader.category_index)

    model = AGGExperiment(
        model_params=config["model_params"],
        optimiser_params=config["optimiser_params"],
        data_params=config["data_params"],
        logging_params=config["logging_params"],
    )

    loss_callback = ModelCheckpoint(
        monitor="val_mse_loss",
        save_top_k=4,
        mode="min",
        filename="model-{epoch:02d}-{val_mse_loss:.6f}",
    )

    callbacks = [
        loss_callback,
    ]

    version_path = f"AGG-pm25-{datetime.now().strftime('%d-%m_%H:%M:%S')}"

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
    )

    pprint(config)
    for key, value in config.items():
        trainer.logger.experiment.add_text(key, str(value), global_step=0)

    log_path = Path(tb_logger.log_dir)
    with open(log_path / "config.yaml", "w") as yml_file:
        yaml.dump(config, yml_file, default_flow_style=False)

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    sys.exit(main())
