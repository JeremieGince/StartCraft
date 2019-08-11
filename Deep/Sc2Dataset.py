import numpy as np
import torch
from torch.utils.data import Dataset
import os


class Sc2Dataset(Dataset):

    def __init__(self, botname, hm_choices, hm_unit_class, action_maker=True, units_creator=False):
        super(Sc2Dataset, self).__init__()

        assert not (action_maker and units_creator) and (action_maker or units_creator)

        self.data_key = "actions_choice_data" if action_maker else "create_units_data"

        self.directory_path = f"training_data/train_data_{botname}_{hm_choices}_choices_{hm_unit_class}_units"
        assert os.path.exists(self.directory_path), f"The dir {self.directory_path} doesn't exist"

        self.data = list()
        self.get_data()
        temp_path = self.directory_path
        self.directory_path = "../" + self.directory_path
        if os.path.exists(self.directory_path):
            self.get_data()
        else:
            self.directory_path = temp_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return [torch.from_numpy(self.data[i][0]), np.argmax(self.data[i][1])]

    # def balance_data(self):
    #     data_counter = {str(d): 0 for d in set(targets)}
    #     data_container = {str(d): [] for d in set(targets)}
    #     for i, tar in enumerate(targets):
    #         data_counter[str(tar)] += 1
    #         data_container[str(tar)].append((ts[i], tar))
    #     min_count = min(data_counter.values())
    #
    #     # print(data_counter)
    #
    #     new_ts = []
    #     for t in data_container.values():
    #         new_ts += [d[0] for d in t][:min_count]
    #     new_tar = []
    #     for t in data_container.values():
    #         new_tar += [d[1] for d in t][:min_count]
    #
    #     return np.array(new_ts), np.array(new_tar)

    def get_data(self):
        for i, filename in enumerate(os.listdir(self.directory_path)):
            game_data = np.load(self.directory_path + "/" + filename, allow_pickle=True).item()
            for data in game_data[self.data_key]:
                [img, tar] = data
                if len(img.shape) < 3:
                    img = img[np.newaxis, :, :]
                if sum(tar) > 1:
                    print(tar)
                self.data.append([img, tar])


if __name__ == "__main__":
    dataset = Sc2Dataset("JarexProtoss", 5, 11)
    print("__len__: ", len(dataset))
    for i in range(3):
        print(f"__getitem__({i}): ", dataset[i])

    # dataset.balance_data()
