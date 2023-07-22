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
from pathlib import Path

import pandas as pd
import yaml

types = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "GCS",
    "HR",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "Temp",
    "Urine",
    "SaO2",
    "FiO2",
    "DiasABP",
    "MAP",
    "SysABP",
    "pH",
    "PaCO2",
    "PaO2",
    "MechVent",
    "BUN",
    "Creatinine",
    "Glucose",
    "HCO3",
    "HCT",
    "Mg",
    "Platelets",
    "K",
    "Na",
    "WBC",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "RespRate",
    "Lactate",
    "Albumin",
    "TroponinT",
    "TroponinI",
    "Cholesterol",
]

unique_ICUType = [1, 2, 3, 4]


def decompose_physionet_data(config: dict, exists_ok: bool = True):
    root_path = Path(config["data_root"])
    train_set_input = root_path / "set-a"
    # val_set_input = root_path / "set-b"
    # test_set_input = root_path / "set-c"
    # train_set_output = root_path / "Outcomes" / "Outcomes-a.txt"
    # val_set_output = root_path / "Outcomes" / "Outcomes-b.txt"
    # test_set_output = root_path / "Outcomes" / "Outcomes-c.txt"
    # train_output_db = pd.read_csv(train_set_output)
    for entry in train_set_input.iterdir():
        entry_db = pd.read_csv(entry)
        entry_db.iloc[0]


if __name__ == "__main__":
    with open("./data/mongo_config.yaml", "r") as f:
        mongo_config: dict = yaml.safe_load(f)
    decompose_physionet_data(mongo_config)
