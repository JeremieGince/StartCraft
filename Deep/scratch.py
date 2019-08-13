import numpy as np
import os


for filename in os.listdir("training_data/train_data_JarexProtoss_5_choices_11_units"):
    try:
        game_data = np.load("training_data/train_data_JarexProtoss_5_choices_11_units" + "/" + filename,
                            allow_pickle=True).item()
        np.save("training_data/train_data_JarexProtoss_5_choices/" + filename, {"data": game_data["actions_choice_data"]})
        # np.save("training_data/train_data_JarexProtoss_11_units/" + filename, {"data": game_data["create_units_data"]})
    except Exception:
        pass