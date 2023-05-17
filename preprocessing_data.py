import sys
from pprint import pprint

import yaml

from Datasets.Beijing.datareader import AirQualityData
from Datasets.Beijing.datareader import decompose_data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        metavar="FILE",
        help="Path to the database config file",
        default="Datasets/Beijing/data/mongo_config.yaml",
    )
    parser.add_argument(
        "--block-size",
        "-b",
        dest="block_size",
        help="length of the data blocks to be loaded/preloaded",
        default=10,
    )
    parser.add_argument(
        "--preprocess", dest="preprocess", help="Generate preprocessing", default=False
    )
    parser.add_argument(
        "--exists-ok",
        dest="exists_ok",
        help="Force overwrite existing preprocessing data",
        default=True,
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        mongo_config: dict = yaml.safe_load(f)
    if args.preprocess:
        decompose_data(mongo_config, args.block_size, exists_ok=args.exists_ok)
    else:
        train_reader = AirQualityData(args.block_size, args.config, version="train")
        test_reader = AirQualityData(args.block_size, args.config, version="test")
        total_length = len(test_reader) + len(train_reader)
        print(
            (
                f"Beijing Dataset with block_size={args.block_size}:\n"
                f"Train dataset has a length of {len(train_reader)} "
                f"representing {len(train_reader)/total_length*100:02.2f}%\n"
                f"Test dataset has a length of {len(test_reader)} "
                f"representing {len(test_reader)/total_length*100:02.2f}%",
            )
        )
        print("Meta data:")
        pprint(train_reader.meta_data)


if __name__ == "__main__":
    sys.exit(main())
