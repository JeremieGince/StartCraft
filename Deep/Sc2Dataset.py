import numpy as np
import torch
from torch.utils.data import Dataset


class Sc2Dataset(Dataset):

    def __init__(self, directory_path):
        super(Sc2Dataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, i):
        pass

    @staticmethod
    def balance_data(ts, targets):
        data_counter = {str(d): 0 for d in set(targets)}
        data_container = {str(d): [] for d in set(targets)}
        for i, tar in enumerate(targets):
            data_counter[str(tar)] += 1
            data_container[str(tar)].append((ts[i], tar))
        min_count = min(data_counter.values())

        # print(data_counter)

        new_ts = []
        for t in data_container.values():
            new_ts += [d[0] for d in t][:min_count]
        new_tar = []
        for t in data_container.values():
            new_tar += [d[1] for d in t][:min_count]

        # print(new_ts[0], new_tar[0])

        return np.array(new_ts), np.array(new_tar)


if __name__ == "__main__":
    pass
