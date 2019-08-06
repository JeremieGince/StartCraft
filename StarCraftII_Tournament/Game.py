import sc2
from sc2.player import Bot, Computer, BotAI
from sc2 import Result
import random
import cv2
import numpy as np
from ComputerBot import ComputerBot


class Game:
    max_char_per_name = 15
    max_char_per_rank = 2
    max_char_per_line = max_char_per_name + max_char_per_rank + 3
    disable_char = 'X'
    cell_size = (75, 150)

    def __init__(self, side, player1_dict=None, player2_dict=None, realtime=True):
        self.realtime = realtime
        if player1_dict is not None:
            self.player1 = player1_dict["player"]
            self.rank1 = player1_dict["rank"]
            self.enable1 = player1_dict["enable"]

        else:
            self.player1 = None
            self.rank1 = None
            self.enable1 = None

        self.player1_color = (255, 255, 255)

        if player2_dict is not None:
            self.player2 = player2_dict["player"]
            self.rank2 = player2_dict["rank"]
            self.enable2 = player2_dict["enable"]

        else:
            self.player2 = None
            self.rank2 = None
            self.enable2 = None

        self.player2_color = (255, 255, 255)

        self.side = side
        assert self.side == "left" or self.side == "right"

    def run(self, debug=False):
        assert self.player1 is not None or self.player2 is not None

        if self.player1 is None:
            return [False, True]
        elif self.player2 is None:
            return [True, False]

        if debug:
            result = [False, False]
            rn_idx = int(random.random() > 0.5)
            result[rn_idx] = True
            # self.rank1 = self.rank1 if result[0] else self.disable_char
            # self.rank2 = self.rank2 if result[1] else self.disable_char
            # print(result, (self.rank1, self.rank2))
            self.player1_color = (0, 255, 0) if result[0] else (0, 0, 255)
            self.player2_color = (0, 255, 0) if result[1] else (0, 0, 255)
            return result

        game_result = self.run_game()

        print("game_result", type(game_result), game_result)
        print("game_result[0]", type(game_result[0]), game_result[0])
        result = [Result.Victory == game_result[0], Result.Victory == game_result[1]]
        if not result[0] and not result[1]:
            rn_idx = int(random.random() > 0.5)
            result[rn_idx] = True
        # print("run return: ", result)
        # self.rank1 = self.rank1 if result[0] else self.disable_char
        # self.rank2 = self.rank2 if result[1] else self.disable_char

        self.player1_color = (0, 255, 0) if result[0] else (0, 0, 255)
        self.player2_color = (0, 255, 0) if result[1] else (0, 0, 255)
        return result

    def run_game(self):

        try:
            player1_is_bot = issubclass(self.player1, BotAI)
        except TypeError:
            player1_is_bot = False

        try:
            player1_is_computer = issubclass(self.player1, ComputerBot)
        except TypeError:
            player1_is_computer = isinstance(self.player1, ComputerBot)

        try:
            player2_is_bot = issubclass(self.player2, BotAI)
        except TypeError:
            player2_is_bot = False

        try:
            player2_is_computer = issubclass(self.player2, ComputerBot)
        except TypeError:
            player2_is_computer = isinstance(self.player2, ComputerBot)

        # print(player1_is_bot, player1_is_computer, player2_is_bot, player2_is_computer)
        assert (player1_is_bot or player1_is_computer) and (player2_is_bot or player2_is_computer)

        if player1_is_computer and player2_is_bot:
            raw_game_result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
                Bot(self.player2.BOTRACE, self.player2(), name=self.player2.BOTNAME),
                Computer(race=self.player1.BOTRACE, difficulty=self.player1.DIFFICULTY)
            ], realtime=self.realtime)
            game_result = [Result.Defeat if raw_game_result == Result.Victory else Result.Victory, raw_game_result]
        elif player1_is_bot and player2_is_computer:
            raw_game_result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
                Bot(self.player1.BOTRACE, self.player1(), name=self.player1.BOTNAME),
                Computer(self.player2.BOTRACE, self.player2.DIFFICULTY)
            ], realtime=self.realtime)
            game_result = [raw_game_result, Result.Defeat if raw_game_result == Result.Victory else Result.Victory]
        elif player1_is_computer and player2_is_computer:
            game_result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
                Computer(self.player1.BOTRACE, self.player1.DIFFICULTY),
                Computer(self.player2.BOTRACE, self.player2.DIFFICULTY)
            ], realtime=self.realtime)
        elif player1_is_bot and player2_is_bot:
            game_result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
                Bot(self.player1.BOTRACE, self.player1(), name=self.player1.BOTNAME),
                Bot(self.player2.BOTRACE, self.player2(), name=self.player2.BOTNAME)
            ], realtime=self.realtime)
        else:
            raise ValueError("players must be ComputerBot or BatAI instance")

        return game_result

    def empty_str(self):
        out = '-' * self.max_char_per_line + '\n'
        if self.side == "left":
            out += '|' + " " * self.max_char_per_rank + '|' + " " * self.max_char_per_name + '|' + '\n'
            out += '|' + " " * self.max_char_per_rank + '|' + " " * self.max_char_per_name + '|' + '\n'
        else:
            out += '|' + " " * self.max_char_per_name + '|' + " " * self.max_char_per_rank + '|' '\n'
            out += '|' + " " * self.max_char_per_name + '|' + " " * self.max_char_per_rank + '|' '\n'

        out += '-' * self.max_char_per_line + '\n'
        return out

    def __str__(self):
        if self.player1 is None and self.player2 is None:
            return self.empty_str()

        out = '-' * self.max_char_per_line + '\n'
        out += self.get_line1(for_str=True)
        out += self.get_line2(for_str=True)
        out += '-' * self.max_char_per_line + '\n'
        return out

    def get_line1(self, for_str=False):
        out = ""

        rank1 = self.rank1

        if for_str and not self.enable1:
            rank1 = self.disable_char

        if self.player1 is None:
            rank1_str = ' ' * self.max_char_per_rank
        elif len(str(rank1)) > self.max_char_per_rank:
            rank1_str = '.' * max(3, self.max_char_per_rank)
        else:
            rank1_str = str(rank1) + " " * (self.max_char_per_rank - len(str(rank1)))

        if self.player1 is None:
            player1_str = ' ' * self.max_char_per_name
        elif len(self.player1.BOTNAME) >= self.max_char_per_name:
            player1_str = ''.join([self.player1.BOTNAME[i] for i in range(self.max_char_per_name)])
        else:
            player1_str = self.player1.BOTNAME + " " * (self.max_char_per_name - len(self.player1.BOTNAME))

        if self.side == "left":
            out += '|' + rank1_str + '|' + player1_str + '|' + '\n'
        else:
            out += '|' + player1_str + '|' + rank1_str + '|' '\n'
        return out

    def get_line2(self, for_str=False):
        out = ""

        rank2 = self.rank2

        if for_str and not self.enable2:
            rank2 = self.disable_char

        if self.player2 is None:
            rank2_str = ' ' * self.max_char_per_rank
        elif len(str(rank2)) > self.max_char_per_rank:
            rank2_str = '.' * max(3, self.max_char_per_rank)
        else:
            rank2_str = str(rank2) + " " * (self.max_char_per_rank - len(str(rank2)))

        if self.player2 is None:
            player2_str = ' ' * self.max_char_per_name
        elif len(self.player2.BOTNAME) >= self.max_char_per_name:
            player2_str = ''.join([self.player2.BOTNAME[i] for i in range(self.max_char_per_name)])
        else:
            player2_str = self.player2.BOTNAME + " " * (self.max_char_per_name - len(self.player2.BOTNAME))

        if self.side == "left":
            out += '|' + rank2_str + '|' + player2_str + '|' + '\n'
        else:
            out += '|' + player2_str + '|' + rank2_str + '|' '\n'
        return out

    def get_img_cell(self):
        img = np.zeros((self.cell_size[0], self.cell_size[1], 3))
        y0, dy = 15, self.cell_size[0]//2 - 2
        font_size = (self.cell_size[0] * 0.30) / 50
        lines = [self.get_line1(), self.get_line2()]
        for i, line in enumerate(lines):
            y = y0 + i*dy
            line = line.replace('\n', '')
            line = line.replace('|', ' ')
            p_color = self.player1_color if i == 0 else self.player2_color
            cv2.putText(img, line, (0, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, p_color, thickness=1)
        cv2.rectangle(img, (0, 0), (self.cell_size[1] - 1, self.cell_size[0] - 1), color=(255, 0, 0), thickness=1)
        return img


if __name__ == '__main__':
    from MarineOverflow import MarineOverflow
    from JarexTerran import JarexTerran
    from JarexProtoss import JarexProtoss

    class Dummy1(MarineOverflow):
        BOTNAME = "Dummy1"

    lgames = list()
    rgames = list()
    game = Game("left", {"player": MarineOverflow, "rank": 1, "enable": True},
                {"player": JarexProtoss, "rank": 12, "enable": True}, realtime=False)
    print(game)

    while True:
        img = game.get_img_cell()
        # img = cv2.resize(img, (500, 250))
        cv2.imshow('test cell', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    print(game.run())

    print(game)

    while True:
        img = game.get_img_cell()
        # img = cv2.resize(img, (500, 250))
        cv2.imshow('test cell', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    rgame = Game("right", {"player": JarexTerran, "rank": 1, "enable": True},
                 {"player": Dummy1, "rank": 'X', "enable": False})
    rgame.player1_color = (0, 255, 0)
    print(rgame)

    while True:
        img = rgame.get_img_cell()
        # img = cv2.resize(img, (500, 250))
        cv2.imshow('test cell', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    empty_game = Game("left")
    print(empty_game)

    # for i in range(1, 6):
    #     lgames.append(Game({"player": MarineOverflow, "rank": i}, {"player": MarineOverflow, "rank": i+5}, "left"))
    # for i in range(6, 11):
    #     rgames.append(Game({"player": MarineOverflow, "rank": i}, {"player": MarineOverflow, "rank": i+5}, "right"))
    #
    # for i in range(len(rgames)):
    #     print(str(lgames[i]) + ' '*10 + str(rgames[i]) + '\n')
