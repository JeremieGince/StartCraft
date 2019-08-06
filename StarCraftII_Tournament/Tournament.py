from StarCraftII_Tournament.Game import Game
import cv2
import random
import numpy as np
import time
import threading
from datetime import datetime


class Tournament:
    left_players = list()
    right_players = list()

    graph = list()
    current_step = 0
    final_step = 1
    nb_layer = 1

    def __init__(self, list_of_left_player, list_of_right_player, realtime=True):
        self.realtime = realtime
        self.nb_player = len(list_of_left_player) + len(list_of_right_player)
        self.final_step = int(np.ceil(np.log2(self.nb_player/2)))
        self.nb_games = -1
        # self.final_step = int(np.ceil(self.nb_player/4))
        print(self.final_step)
        print(np.log2(self.nb_player/2))
        print(self.nb_player/4)
        self.graph = [None for _ in range(self.final_step+1)]

        self.classification(list_of_left_player, list_of_right_player)

        self.create_tournament_graph()

        for step in self.graph:
            for _ in step["left_layer"]["games"] + step["right_layer"]["games"]:
                self.nb_games += 1
        print("nb_games: ", self.nb_games)

        print(str(self))
        self.show_th = threading.Thread(target=self.show, daemon=True)
        # self.show()
        self.show_th.start()

    def create_tournament_graph(self):
        for i in range(self.current_step, self.final_step+1):
            # print(i, self.graph)
            self.graph[i] = self.create_tournament_layer(i)
            # print(self.str_layer(self.graph[i]))

    def create_tournament_layer(self, step):
        if step < self.current_step:
            return self.graph[step]
        elif step > self.current_step:
            # print("step: ", step, ' ', self.nb_player / (2**(step+1)))
            n = int(np.ceil(self.nb_player / (2**(step+1))))
            left_layer = self.create_side_layer(n, "left")
            right_layer = self.create_side_layer(n, "right")
        elif not step:
            left_layer = self.create_side_layer(self.left_players, "left")
            right_layer = self.create_side_layer(self.right_players, "right")
        elif step == self.final_step:
            finals_players = self.get_enable_players(self.left_players)+self.get_enable_players(self.right_players)
            left_layer = self.create_side_layer(finals_players, "left")
            right_layer = None
        else:
            left_layer = self.create_side_layer(self.get_enable_players(self.left_players), "left")
            right_layer = self.create_side_layer(self.get_enable_players(self.right_players), "right")
        layer = {"step": step, "left_layer": left_layer, "right_layer": right_layer, "done": False}
        return layer

    def create_side_layer(self, side_players, side):
        # Il faut gerer les nombres de players impaire de façon à ce que le meilleur fasse plus de game
        assert isinstance(side_players, int) or isinstance(side_players, list)

        side_layer = {"games": list(), "side": side}
        if isinstance(side_players, int):
            for i in range(side_players//2):
                side_layer["games"].append(Game(side, None, None, self.realtime))
            if side_players % 2:
                # print("we had an extra game for hod player")
                side_layer["games"].append(Game(side, None, None, self.realtime))
        else:
            side_ranks = [player["rank"] for player in side_players]

            while len(side_ranks):
                max_rank = max(side_ranks)
                min_rank = min(side_ranks)

                if max_rank != min_rank:
                    side_layer["games"].append(Game(side,
                                                    self.get_player_by_rank(max_rank),
                                                    self.get_player_by_rank(min_rank),
                                                    self.realtime))
                    side_ranks.remove(max_rank)
                    side_ranks.remove(min_rank)
                else:
                    side_layer["games"].append(Game(side,
                                                    self.get_player_by_rank(max_rank),
                                                    None,
                                                    self.realtime))
                    side_ranks.remove(max_rank)

        return side_layer

    def get_player_by_rank(self, rank):
        player = None
        for p in self.left_players+self.right_players:
            if p["rank"] == rank:
                player = p
                break
        return player

    def get_player_by_addr(self, addr):
        player = None
        for p in self.left_players+self.right_players:
            if p["player"] == addr:
                player = p
                break
        return player

    def get_player_by_unique_attribute(self, attribute, value):
        player = None
        for p in self.left_players + self.right_players:
            if p[attribute] == value:
                player = p
                break
        return player

    def get_enable_players(self, players):
        enable_players = list()
        for p in players:
            if p["enable"]:
                enable_players.append(p)
        return enable_players

    def run(self, debug=False):
        for step in self.graph:
            self.run_layer(step, debug)

    def run_layer(self, step, debug=False):
        if step["step"] == self.final_step:
            return self.run_final_layer(step, debug)

        time.sleep(3)

        for game in step["left_layer"]["games"] + step["right_layer"]["games"]:
            result = game.run(debug)
            if game.player1 is not None:
                self.get_player_by_addr(game.player1)["enable"] = result[0]
            if game.player2 is not None:
                self.get_player_by_addr(game.player2)["enable"] = result[1]
            print(self)
            time.sleep(3)

        self.current_step += 1
        self.create_tournament_graph()

    def run_final_layer(self, layer, debug=False):
        time.sleep(3)
        game = layer["left_layer"]["games"][0]
        result = game.run(debug)
        self.get_player_by_addr(game.player1)["enable"] = result[0]
        self.get_player_by_addr(game.player2)["enable"] = result[1]
        # self.create_tournament_graph()
        # print(self.left_players)
        print(self)

        # time.sleep(5)
        self.current_step += 1
        self.create_tournament_graph()
        winner = self.get_enable_players(self.left_players+self.right_players)[0]
        print(f"The winner is {winner['player'].BOTNAME} with rank: {winner['rank']}")
        time.sleep(10)

    def __str__(self):
        str_layers = [self.str_layer(layer) for layer in self.graph]
        this = str_layers[0]
        start_coord = (2 if len(str_layers) > 2 else 0, Game.max_char_per_line + 2)
        for i, layer in enumerate(str_layers):
            if not i:
                continue

            # print("this: ")
            # print(this)-
            # print("layer: ")
            # print(layer)
            # print("***---***")

            this_split = this.split('\n')
            layer_split = layer.split('\n')

            for line in range(len(layer_split)):
                this_line = list(this_split[start_coord[0] + line])
                # print(len(this_line), ''.join(this_line), this_line)
                # print(len(layer_split[line]), layer_split[line])
                for col in range(len(layer_split[line])):
                    this_line[start_coord[1] + col] = layer_split[line][col]

                this_split[start_coord[0] + line] = ''.join(this_line)

            start_coord = (start_coord[0] + 2 if i+1 != len(str_layers)-1 else start_coord[0],
                           start_coord[1]+Game.max_char_per_line+2)
            this = '\n'.join(this_split)
        return this

    def str_layer(self, layer):
        if layer["step"] == self.final_step:
            return self.str_final_layer()

        layer_str = ""
        left_layer = layer["left_layer"]
        right_layer = layer["right_layer"]

        spacing_between_sides = (2*Game.max_char_per_line + 4) * (self.final_step-layer["step"]-1) \
                                + Game.max_char_per_line + 4

        max_nb_games = max(len(left_layer["games"]), len(right_layer["games"]))
        for i in range(max_nb_games):
            lgame = left_layer["games"][i]
            rgame = right_layer["games"][i]

            lgame_split = str(lgame).split('\n')
            rgame_split = str(rgame).split('\n')

            # lgame_split.remove('')
            # rgame_split.remove('')

            max_line = max(len(lgame_split), len(rgame_split))
            for line in range(max_line):
                if lgame_split[line] == '':
                    lgame_split[line] = ' ' * Game.max_char_per_line
                if rgame_split[line] == '':
                    rgame_split[line] = ' ' * Game.max_char_per_line

                layer_str += lgame_split[line] + ' ' * spacing_between_sides + rgame_split[line] + '\n'

        return layer_str

    def str_final_layer(self):
        layer_str = ""
        left_layer = self.graph[-1]["left_layer"]

        lgame = left_layer["games"][0]

        lgame_split = str(lgame).split('\n')

        for line in range(len(lgame_split)):
            layer_str += lgame_split[line] + '\n'

        return layer_str

    def show(self):
        while True:
            # img = np.zeros((750, 850, 3))
            img = cv2.imread("sc2_tournament_background.jpg")
            cv2.putText(img, "StarCraft II Tournament", (50, int(img.shape[1]/10)), cv2.FONT_HERSHEY_SIMPLEX, 5,
                        (0, 0, 255), thickness=3)
            cv2.putText(img, f"{datetime.today().strftime('%Y %m %d')}", (600, int(img.shape[1] / 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness=3)
            # print(str(self))
            # text = str(self)
            # y0, dy = 500, 10
            # x0, dx = 50, 10
            # for i, line in enumerate(text.split('\n')):
            #     y = y0 + i * dy
            #     for j, col in enumerate(line.split(' ')):
            #         x = x0 + j * dx
            #         cv2.putText(img, col, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)
            # cv2.addText(img, str(self), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255))
            # img = cv2.resize(img, (850, 750))
            img = self.put_graph_img(img)
            cv2.imshow("Tournament", img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            time.sleep(0.01)

    def put_graph_img(self, screen):
        padx = int((screen.shape[1] - (2*self.final_step + 2) * Game.cell_size[1]) / (2*self.final_step + 2))
        # padx = 1

        x0, dx = int(padx), int(Game.cell_size[1] + padx)

        nb_games_first_left_layer = sum([1 for _ in self.graph[0]["left_layer"]["games"]])
        y0 = int(screen.shape[0] / 3)
        pady = int((screen.shape[0] - y0 - nb_games_first_left_layer * Game.cell_size[0]) / nb_games_first_left_layer)
        dy0, dy = int((Game.cell_size[0] + pady) / 2), int(Game.cell_size[0] + pady)

        # print(
        #     f"pad: {padx, pady}, screen.shape: {screen.shape}, step: {self.final_step}, Game.cell_size: {Game.cell_size}")

        for li, step in enumerate(self.graph):
            left_layer = step["left_layer"]
            for gi, game in enumerate(left_layer["games"]):
                x = x0 + li * dx
                y = (y0 + li * dy0 + gi * dy) if li != len(self.graph) - 1 else (y0 + (li-1) * dy0 + gi * dy)

                cell = game.get_img_cell()
                screen[y:y+cell.shape[0], x:x+cell.shape[1]] = cell

        x0 = int(screen.shape[1] - x0 - Game.cell_size[1])
        dx = -dx

        for li, step in enumerate(self.graph):
            if li == len(self.graph) - 1:  # final step doesn't have right layer
                continue
            right_layer = step["right_layer"]
            for gi, game in enumerate(right_layer["games"]):
                x = x0 + li * dx
                y = y0 + li * dy0 + gi * dy

                cell = game.get_img_cell()
                screen[y:y + cell.shape[0], x:x + cell.shape[1]] = cell

        return screen

    @staticmethod
    def sort_players_by_rank(mydict):
        sorted_dict = dict()
        for key, value in sorted(mydict.items(), key=lambda item: item[1]["rank"]):
            sorted_dict[key] = value
        return sorted_dict

    def classification(self, list_of_left_player, list_of_right_player):
        for i, player in enumerate(list_of_left_player):
            self.left_players.append({"player": player, "enable": True, "rank": i})

        for i, player in enumerate(list_of_right_player):
            rank = i + len(list_of_left_player)
            self.right_players.append({"player": player, "enable": True, "rank": rank})

        ranks = list(range(1, self.nb_player+1))
        random.shuffle(ranks)
        # print(ranks)
        for i, p in enumerate(self.left_players+self.right_players):
            p["rank"] = ranks[i]


if __name__ == '__main__':
    from MarineOverflow import MarineOverflow
    from JarexTerran import JarexTerran
    from JarexProtoss import JarexProtoss
    from ComputerBot import ComputerBot
    from sc2 import Race, Difficulty, BotAI

    class Dummy1(MarineOverflow):
        BOTNAME = "Dummy1"

    class Dummy2(JarexProtoss):
        BOTNAME = "Dummy2"

    class Dummy3(MarineOverflow):
        BOTNAME = "Dummy3"

    class Dummy4(JarexTerran):
        BOTNAME = "Dummy4"

    class Dummy5(MarineOverflow):
        BOTNAME = "Dummy5"

    class Dummy6(JarexProtoss):
        BOTNAME = "Dummy6"

    class Dummy7(MarineOverflow):
        BOTNAME = "Dummy7"

    class Dummy8(JarexTerran):
        BOTNAME = "Dummy8"

    class Dummy9(MarineOverflow):
        BOTNAME = "Dummy9"

    class Dummy10(JarexProtoss):
        BOTNAME = "Dummy10"

    class Dummy11(MarineOverflow):
        BOTNAME = "Dummy11"

    class Dummy12(JarexTerran):
        BOTNAME = "Dummy12"

    class Dummy13(MarineOverflow):
        BOTNAME = "Dummy13"

    class Dummy14(JarexProtoss):
        BOTNAME = "Dummy14"


    # left = [MarineOverflow, Dummy1, Dummy3, Dummy5, Dummy7, Dummy9, Dummy11, Dummy13]
    # right = [JarexTerran, Dummy2, Dummy4, Dummy6, Dummy8, Dummy10, Dummy12, Dummy14]

    # left = [MarineOverflow, Dummy1, Dummy3, Dummy5, Dummy7, Dummy9, Dummy11]
    # right = [JarexTerran, Dummy2, Dummy4, Dummy6, Dummy8, Dummy10, Dummy12]

    # left = [MarineOverflow, Dummy1, Dummy3, Dummy5, Dummy7, Dummy9]
    # right = [JarexTerran, Dummy2, Dummy4, Dummy6, Dummy8, Dummy10]

    # #         Gince,      Remi,   Richard, Gab,   LP
    left = [MarineOverflow, Dummy1, Dummy3, Dummy5, Dummy7]

    right = [JarexTerran, JarexProtoss, ComputerBot("TerranCom", Race.Terran, Difficulty.Medium), Dummy2, Dummy4]

    # left = [MarineOverflow, Dummy1, Dummy3, Dummy5]
    # right = [JarexTerran, Dummy2, Dummy4, Dummy6]

    # print(issubclass(MarineOverflow, BotAI), issubclass(ComputerBot("Com1", Race.Zerg, Difficulty.Easy), ComputerBot))

    # left = [MarineOverflow, ComputerBot("Com0", Race.Zerg, Difficulty.Hard)]
    # right = [JarexProtoss, JarexTerran]

    # left = [MarineOverflow, ComputerBot]
    # right = [JarexProtoss, ComputerBot]

    # left = [MarineOverflow]
    # right = [JarexTerran]

    tournament = Tournament(left, right, realtime=False)
    tournament.run(debug=False)
