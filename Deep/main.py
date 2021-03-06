import random

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

from JarexProtoss import JarexProtoss
from Deep.Sc2Dataset import Sc2Dataset
from Deep.Models import Sc2Net, Sc2UnitMakerNet
from Deep.ModelTrainer import ModelTrainer
import torch

if __name__ == '__main__':
    hm_win = 10
    max_game = 300
    hm_win_per_train = 10
    win_counter = 0
    last_win_update = 0
    game_counter = 0
    difficulty = Difficulty.Harder
    gamma = 1/(hm_win/hm_win_per_train)
    epsilon = 0.5

    races = [Race.Zerg, Race.Terran, Race.Protoss]
    ennemie_is_stats_model = True

    while win_counter < hm_win:

        result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
            Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=True, human_control=False,
                                                   debug=False, take_training_data=True, epsilon=epsilon)),

            Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=False, human_control=False,
                                                   debug=False, take_training_data=True, epsilon=1.0))
            if ennemie_is_stats_model
            else Computer(random.choice(races), difficulty)

        ], realtime=False)

        if result == sc2.Result.Victory:
            win_counter += 1
        game_counter += 1

        print(f"-" * 175)
        print(f"win_counter: {win_counter}, game_counter: {game_counter}")
        print(f"-" * 175)

        if not win_counter % hm_win_per_train and win_counter and win_counter != last_win_update:

            # Training of action maker model
            dataset = Sc2Dataset("JarexProtoss", 5, 11, action_maker=True, units_creator=False)
            model = Sc2Net(input_chanels=1, output_size=5)
            model_trainer = ModelTrainer(model=model, dataset=dataset)
            model_trainer.train(15)
            model_trainer.save_model(filename="../Models/JarexProtoss_action_model.pth", difficulty=difficulty)
            # torch.save(model, "../Models/JarexProtoss_action_model.pth")
            # model_trainer.plot_history()

            # Training of unit maker model
            dataset = Sc2Dataset("JarexProtoss", 5, 11, action_maker=False, units_creator=True)
            model = Sc2UnitMakerNet("JarexProtoss")
            model.train(dataset, max_epoch=100, verbose=False)
            model.save()
            # model.plot_history()

            last_win_update = win_counter
            epsilon *= gamma

        if game_counter == max_game:
            break

    print(f"--- Training on {difficulty} Finished ---")
    print(f"win_counter: {win_counter}, game_counter: {game_counter}")
    print(f"ratio of game win: {win_counter / game_counter}")
    print(f"-"*175)
