import numpy as np


def random_index(data_len: int, sparsity: float):
    idx = np.arange(0, data_len)
    subset_size = int(np.floor(data_len) * sparsity)
    np.random.shuffle(idx)
    removed = idx[:subset_size]
    remainder = idx[subset_size:]
    removed.sort()
    remainder.sort()
    return removed, remainder
