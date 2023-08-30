import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from pymongo import MongoClient

from AGG.transformer_model import AsynchronousGraphGeneratorTransformer
from Datasets.Beijing.datareader import features
from Datasets.Beijing.datareader import graph_template
from Datasets.Beijing.datareader import normalise
from Datasets.Beijing.datareader import target_template
from Datasets.Beijing.datareader import test_masks

best_model_path = Path()
assert best_model_path.exists()
checkpoint = torch.load(best_model_path)

model_state_dict = checkpoint["state_dict"]
config_path = best_model_path.parent.parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

if "type" in config["model_params"]:
    config["model_params"].pop("type")

agg = AsynchronousGraphGeneratorTransformer(**config["model_params"])
agg.load_state_dict(model_state_dict)

db_config_path = Path(config["data_params"]["db_config"])
with open(db_config_path) as f:
    db_config = yaml.safe_load(f)
mongo_db_client = MongoClient(host=db_config["host"], port=db_config["port"])
db = mongo_db_client[db_config["base"]]
db_params = db["param"]
params = db_params.find_one({})
db_raw = db["raw"]
block_name = f'block_{config["data_params"]["block_size"]:02d}_{100 * config["data_params"]["sparsity"]}%_pm25'
index_entry = db[block_name]["indexes"]["test"]

raw_test = list(
    db_raw.find({"time": {"$gt": test_masks[0][0], "$lt": test_masks[0][1]}}).sort(
        "time"
    )
)
idx = np.arange(0, len(raw_test))
np.random.shuffle(idx)
subset_size = idx.shape[0] // 2
removed = idx[:subset_size]
remainder = idx[subset_size:]
removed.sort()
remainder.sort()
samples: list = []
time_scale: int = 3600 * 72
block_size = config["data_params"]["block_size"]
write_data = []
for n in range(0, len(remainder) - block_size, block_size):
    input_index = remainder[n : (n + block_size)]
    max_time = raw_test[input_index[-1]]["time"]
    graph = copy.deepcopy(graph_template)
    for k in range(input_index.shape[0]):
        raw_entry = raw_test[input_index[k]]
        graph["node_features"].append(
            normalise(
                raw_entry["node_features"],
                params["scaling"][features[raw_entry["type_index"]]],
            )
        )
        graph["time"].append(
            (max_time - raw_entry["time"]).total_seconds() / time_scale
        )
        graph["category_index"].append(raw_entry["category_index"])
        graph["type_index"].append(raw_entry["type_index"])
        graph["spatial_index"].append(raw_entry["spatial_index"])
    graph["key_padding_mask"] = (np.zeros_like(input_index) != 0).tolist()
    mask = (removed > min(input_index)) & (removed < max(input_index))
    test_target = removed[mask]
    for i in range(test_target.shape[0]):
        k = test_target[i]
        graph_sample = copy.deepcopy(graph)
        target = copy.deepcopy(target_template)
        if raw_test[k]["type_index"] == 0:
            feature_type = features[raw_test[k]["type_index"]]
            target["features"] = [
                normalise(
                    raw_test[k]["node_features"], params["scaling"][feature_type]
                ),
            ]
            target["type_index"] = [
                raw_test[k]["type_index"],
            ]
            target["spatial_index"] = [
                raw_test[k]["spatial_index"],
            ]
            target["category_index"] = [
                raw_test[k]["category_index"],
            ]
            target["time"] = [
                (max_time - raw_test[k]["time"]).total_seconds() / time_scale,
            ]
            graph_sample["target"] = target
            graph_sample["datetime"] = raw_test[k]["time"]
            write_data.append(graph_sample)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mini_batch_size = 20
# agg = agg.to(device)
# agg.eval()
# pm25_list = []
# error_list = []
# time_list = []
# ground_truth = []
# spatial_index = []
# with torch.no_grad():
#     for mini_batch_idx in range(0, len(samples), mini_batch_size):
#         if mini_batch_idx + 10 < len(samples):
#             mini_batch = samples[mini_batch_idx:(mini_batch_idx + mini_batch_size)]
#         else:
#             mini_batch = samples[mini_batch_idx:]
#         graph = collate_graph_samples(mini_batch)
#         y_hat, attention_list = agg(graph, device=device)
#         pm25_list.append(y_hat.to('cpu'))
#         ground_truth.append(graph.target.features)
#         spatial_index.append(graph.target.spatial_index)
#         error_list.append(torch.abs(graph.target.features.to(device) - y_hat).to('cpu'))
#         time_list.append(graph.target.time)
#
#     time = torch.cat(time_list, dim=0).flatten().numpy()
#     pm25_est = torch.cat(pm25_list, dim=0).flatten().numpy()
#     error = torch.cat(error_list, dim=0).flatten().numpy()
#     ground_truth = torch.cat(ground_truth, dim=0).flatten().numpy()
#     spatial_index = torch.cat(spatial_index, dim=0).flatten().numpy()
#     sample_mask = sample.type_index == 0
#     sample_pm25 = sample.node_features[sample_mask].numpy()
#     sample_time = sample.time[sample_mask]
#     sample_max = sample_time.max()
#     sample_time = (sample_max - sample_time).numpy()
#     sample_spatial = sample.spatial_index[sample_mask].numpy()
#     time = sample_max - time
#
# plt.figure()
# location = 0
# slice = spatial_index == location
# plt.plot(time[slice], ground_truth[slice], 'xk', label='ground truth')
# plt.plot(time[slice], pm25_est[slice], '.r', label='AGG estimate')
# plt.plot(sample_time[sample_spatial == location], sample_pm25[sample_spatial == location], '--go', label='input pm25')
# plt.legend()
# plt.show()
