from datetime import datetime
from pathlib import Path
from typing import Any
from typing import List

import pandas as pd

data_path = Path("")
assert data_path.exists(), f"{data_path} does not exist"


def decompose_pm2_5_data(file: Path):
    decomposed_data = []
    header = []
    with open(file, "r", encoding="utf-8") as f:
        contents = f.readlines()
        for i, line in enumerate(contents):
            strings = line[:-1].split(",")
            if i == 0:
                for title in strings:
                    title = title.split('"')[1]
                    if title not in ["year", "month", "day", "hour"]:
                        header.append(title)
                    else:
                        if title == "year":
                            header.append("datetime")
            else:
                data: List[Any] = []
                idx = 0
                for data_field in header:
                    if data_field == "datetime":
                        idx = 5
                        data.append(
                            datetime(
                                year=int(strings[1]),
                                month=int(strings[2]),
                                day=int(strings[3]),
                                hour=int(strings[4]),
                            )
                        )
                    else:
                        if strings[idx] == "NA":
                            data.append(None)
                        else:
                            if idx in [15, 17]:
                                data.append(strings[idx].split('"')[1])
                            else:
                                data.append(float(strings[idx]))
                        idx += 1
                decomposed_data.append(data)
    return header, decomposed_data


datasets = []
for file in data_path.iterdir():
    if str(file)[-4:] == ".csv":
        header, pm2_5_data = decompose_pm2_5_data(file)
        df = pd.DataFrame(pm2_5_data, columns=header)
        datasets.append(df)
df_concat = pd.concat(datasets)
df_concat.sort_values(by=["datetime", "station"], inplace=True)
df_concat.reset_index(inplace=True, drop=True)
df_concat["wd"] = df_concat["wd"].astype(str)
df_concat["station"] = df_concat["station"].astype(str)
df_concat.to_hdf(data_path / "pm2_5_df.h5", key="df")
