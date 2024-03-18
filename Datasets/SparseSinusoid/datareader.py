from torch.utils.data import Dataset
import numpy as np
import copy
from tqdm import tqdm

from Datasets.data_tools import random_index
from AGG.extended_typing import ContinuousTimeGraphSample


class SinusoidData(object):
    def __init__(
            self,
            fs: int = 1000,
            f: float = 1.0,
            t_end: int = 60,
            sparsity: float = 0.98,
            context_length: int = 50,
            time_scale: float | None = None
    ):
        t = np.arange(0, t_end, 1 / fs)
        x = np.sin(2 * np.pi * f * t)
        self.fs = fs
        self.f = f
        self.x = x
        self.t = t
        t = np.arange(0, t_end, 1 / fs)
        x = np.sin(2 * np.pi * f * t)
        self.x = x
        self.t = t
        removed, remainder = random_index(x.shape[0], sparsity)
        self.training_samples = x[remainder]
        self.training_samples_t = t[remainder]
        _, target_index = random_index(removed.shape[0], sparsity)
        self.target_samples = x[removed[target_index]]
        self.target_samples_t = t[removed[target_index]]
        self.context_length = context_length
        if time_scale is None:
            self.time_scale = self.get_max_time()
        else:
            self.time_scale = time_scale
        self.data_samples, self.ground_truth = self.construct_data_samples()

    def get_max_time(self) -> float:
        max_time = 0
        for i in range(self.training_samples.shape[0] - self.context_length):
            sample = self.training_samples_t[i:i + self.context_length]
            sample_max = sample.max() - sample.min()
            if sample_max > max_time:
                max_time = sample_max
        return max_time

    def construct_data_samples(self) -> (list[ContinuousTimeGraphSample], list[dict]):
        data_samples = []
        gt_samples = []
        input_graph = {}
        for i in tqdm(range(self.training_samples.shape[0] - self.context_length)):
            input_graph["node_features"] = self.training_samples[i:i + self.context_length].tolist()
            time_sample = self.training_samples_t[i:i + self.context_length]
            gt_graph = {
                "features": self.training_samples[i + self.context_length:i + self.context_length + 1],
                "time": time_sample
            }
            time_sample_bias = time_sample.min()
            input_graph["time"] = ((time_sample - time_sample_bias)/self.time_scale).tolist()
            target_mask = np.logical_and(
                self.target_samples_t >= self.training_samples_t[i:i + self.context_length].min(),
                self.target_samples_t <= self.training_samples_t[i:i + self.context_length].max()
            )
            target_inputs = self.target_samples[target_mask]
            target_times = self.target_samples_t[target_mask]
            for j in range(target_inputs.shape[0]):
                target_graph = {
                    "features": target_inputs[j:j + 1].tolist(),
                    "time": ((target_times[j:j + 1] - time_sample_bias)/self.time_scale).tolist()
                }
                graph_sample = copy.deepcopy(input_graph)
                gt_graph_sample = copy.deepcopy(gt_graph)
                gt_graph_sample['target'] = target_inputs[j:j + 1]
                gt_graph_sample['time'] = target_times[j:j + 1]
                gt_samples.append(gt_graph_sample)
                graph_sample['target'] = target_graph
                graph_sample['attention_mask'] = np.ones((self.context_length, self.context_length)).tolist()
                data_samples.append(ContinuousTimeGraphSample(**graph_sample))
        return data_samples, gt_samples

    def plot_training_samples(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 5))
        plt.plot(self.t, self.x, label='sinusoid')
        plt.plot(self.training_samples_t, self.training_samples, 'o',
                 label=f'training samples: {self.training_samples.shape[0]}')
        plt.plot(self.target_samples_t, self.target_samples, 'o',
                 label=f'target samples: {self.target_samples.shape[0]}')
        plt.legend()
        plt.show()


class SinusoidDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            train_split: float = 0.8,
            sinusoid_data: SinusoidData = None,
    ):
        if sinusoid_data is None:
            self.source = SinusoidData()
        else:
            self.source = sinusoid_data
        if train:
            self.data_samples = self.source.data_samples[:int(len(self.source.data_samples) * train_split)]
            self.ground_truth = self.source.ground_truth[:int(len(self.source.ground_truth) * train_split)]
        else:
            self.data_samples = self.source.data_samples[int(len(self.source.data_samples) * train_split):]
            self.ground_truth = self.source.ground_truth[int(len(self.source.ground_truth) * train_split):]

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]


if __name__ == '__main__':
    data = SinusoidData(t_end=60, sparsity=0.98)
    data_reader = SinusoidDataset(sinusoid_data=data)
    print(len(data_reader))
    data.plot_training_samples()
