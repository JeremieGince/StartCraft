import sc2
from sc2 import Difficulty
from sc2 import Race


class ComputerBot:
    BOTNAME = "Computer"
    BOTRACE = Race.Zerg
    DIFFICULTY = Difficulty.Easy

    def __init__(self, name, race, difficulty):
        assert isinstance(race, Race) and isinstance(difficulty, Difficulty) and isinstance(name, str)

        self.BOTNAME = name
        self.BOTRACE = race
        self.DIFFICULTY = difficulty

    def __call__(self):
        return self.DIFFICULTY
