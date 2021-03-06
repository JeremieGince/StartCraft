import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
import warnings

warnings.filterwarnings("ignore")
np.seterr(divide='ignore', invalid='ignore')


class Sc2Dataset(Dataset):

    def __init__(self, botname, hm_choices: int = 0, hm_unit_class: int = 0, action_maker=True, units_creator=False):
        super(Sc2Dataset, self).__init__()

        assert not (action_maker and units_creator) and (action_maker or units_creator)

        if action_maker:
            self.directory_path = f"training_data/train_data_{botname}_{hm_choices}_choices"
        elif units_creator:
            self.directory_path = f"training_data/train_data_{botname}_{hm_unit_class}_units"
        else:
            raise ValueError
        assert os.path.exists(self.directory_path), f"The dir {self.directory_path} doesn't exist"

        self.full_data = list()
        self.get_data()
        temp_path = self.directory_path
        self.directory_path = "../" + self.directory_path
        if os.path.exists(self.directory_path):
            self.get_data()
        else:
            self.directory_path = temp_path

        if action_maker:
            self.format_img_data()

        self.data = list()
        self.balance_data()
        random.shuffle(self.data)

    def __len__(self):
        return min(10_000, len(self.data))

    def __getitem__(self, i):
        return [torch.from_numpy(self.data[i][0]), np.argmax(self.data[i][1])]

    def balance_data(self):
        random.shuffle(self.full_data)
        data_counter = dict()
        for d in self.full_data:
            key = np.argmax(d[1])
            if key in data_counter:
                data_counter[key]["data"].append(d)
                data_counter[key]["counter"] += 1
            else:
                data_counter[key] = {"data": [d], "counter": 1}

        counters = [info["counter"] for _, info in data_counter.items()]
        mean_counter = int(np.mean(counters))
        min_counter = min(counters)
        max_counter = max(counters)

        counters.remove(max_counter)
        second_max_counter = max(counters)

        # print(min_counter, mean_counter, second_max_counter, max_counter)

        for key, info in data_counter.items():
            self.data.extend(info["data"][:mean_counter])

        random.shuffle(self.data)

    def get_data(self):
        for _, filename in enumerate(os.listdir(self.directory_path)):
            game_data = np.load(self.directory_path + "/" + filename, allow_pickle=True).item()
            for data in game_data["data"]:
                self.full_data.append(data)

    def format_img_data(self):
        new_full_data = list()
        for [img, tar] in self.full_data:
            if len(img.shape) < 3:
                img = img[np.newaxis, :, :]
            if sum(tar) > 1:
                print(tar)
            new_full_data.append([img, tar])
        self.full_data = new_full_data


if __name__ == "__main__":
    dataset = Sc2Dataset("JarexProtoss", 5, 11, action_maker=False, units_creator=True)
    print("__len__: ", len(dataset))
    for i in range(3):
        print(f"__getitem__({i}): ", dataset[i])

    # dataset.balance_data()
