import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer

from StarcraftII_Tournament.TutoPlayer.TutoBotPrime.TutoBot import TutoBot


class TutoBotPrime(TutoBot):
    BOTNAME = "TutoBotPrime"


if __name__ == '__main__':
    run_game(maps.get("AbyssalReefLE"), [
        Bot(TutoBot.BOTRACE, TutoBot()),
        Bot(TutoBotPrime.BOTRACE, TutoBotPrime())
    ], realtime=False)
