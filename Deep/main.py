import random

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer

from JarexProtoss import JarexProtoss

if __name__ == '__main__':
    hm_game = 10

    races = [Race.Zerg, Race.Terran, Race.Protoss]
    difficulties = [Difficulty.Medium, Difficulty.Hard]

    for _ in range(hm_game):
        sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
            Bot(JarexProtoss.BOTRACE, JarexProtoss(human_control=False, debug=False)),
            Computer(random.choice(races), random.choice(difficulties))
        ], realtime=False)
