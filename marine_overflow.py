from Jarex_terran import JarexTerran

import random

import sc2
from sc2 import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.player import Bot, Computer
from sc2.helpers import ControlGroup
from sc2.units import Units
import numpy as np


class MarineOverflow(JarexTerran):
    MAX_WORKERS_COUNT = 50
    AVG_WORKERS_PER_CMDC = 16
    MIN_ARMY_SIZE_FOR_ATTACK = 50
    RATIO_DEF_ATT_UNITS = 0.0

    MILITARY_UNIT_CLASS = {UnitTypeId.MARINE: {"max": 200, "priority": 1, "maker_class": UnitTypeId.BARRACKS, "supply": 1}}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.BARRACKS: {"priority": 1, "max": 10,
                                                      "avg_per_cmdc": 3, "add_on": [],
                                                      "upgrade": []},
                                UnitTypeId.MISSILETURRET: {"priority": 100, "max": 25,
                                                           "avg_per_cmdc": 3, "add_on": [], "upgrade": []},
                                UnitTypeId.FACTORY: {"priority": 1, "max": 1,
                                                     "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.ARMORY: {"priority": 1, "max": 1,
                                                    "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.ENGINEERINGBAY: {"priority": 1, "max": 1,
                                                            "avg_per_cmdc": 1, "add_on": [],
                                                            "upgrade": [UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                                                                        UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                                                                        UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL3]},
                                UnitTypeId.BARRACKSTECHLAB: {"priority": 1, "max": 0,
                                                             "avg_per_cmdc": 0, "add_on": [],
                                                             "upgrade": [UpgradeId.COMBATSHIELD]}
                                }

    do_something_after = 0

    def __init__(self, use_model=False):
        super(MarineOverflow, self).__init__()
        self.use_model = use_model

        # faire le scout group, faire un trucs que c'est moi qui fait les choix pendant la game ou random ou par models.
        # faire l'affichage de données, finir d'écouter les tuto pour avoir plus d'idées

    def on_end(self, game_result):
        JarexTerran.on_end(self, game_result)
        # print("Ennemy killed: ", self._game_info.killed_enemy)

    def attack(self):
        # if len(self.attack_group) >= self.MIN_ARMY_SIZE_FOR_ATTACK or self.supply_used >= self.MAX_SUPPLY - 10:
        #     for unit in self.attack_group.idle:
        #         self.current_actions.append(unit.attack(self.find_target(unit, self.state)))
        # elif len(self.known_enemy_units) > 0:
        #     for unit in self.attack_group.idle:
        #         self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))
        # else:
        #     if self.attack_group.amount and self.attack_group.random.distance_to(self.attack_group.center) > 50:
        #         for unit in self.attack_group.idle:
        #             self.current_actions.append(unit.move(self.attack_group.center))

        target = False
        if len(self.attack_group) >= self.MIN_ARMY_SIZE_FOR_ATTACK and self.iteration > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
                # print('prediction: ',choice)

                choice_dict = {0: "No Attack!",
                               1: "Attack close to our nexus!",
                               2: "Attack Enemy Structure!",
                               3: "Attack Eneemy Start!"}

                print("Choice #{}:{}".format(choice, choice_dict[choice]))

            else:
                choice = random.randrange(0, 4)

            if choice == 0:
                # no attack, defend
                wait = random.randrange(20, 165)
                self.do_something_after = self.iteration + wait

            elif choice == 1:
                # attack_closest_unit
                if len(self.known_enemy_units) > 0:
                    # target = self.known_enemy_units.closest_to(self.start_location)
                    # for unit in self.attack_group.idle:
                    #     self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))
                    self.defend(units=self.attack_group)

            elif choice == 2:
                # attack enemy structures
                if len(self.known_enemy_structures) > 0:
                    target = random.choice(self.known_enemy_structures)

            elif choice == 3:
                # attack_enemy_start
                target = self.enemy_start_locations[0]

            if target:
                for unit in self.attack_group.idle:
                    self.current_actions.append(unit.attack(target))

            # wait = random.randrange(20, 165)
            # self.do_something_after = self.iteration + wait

            y = np.zeros(4)
            y[choice] = 1
            # print(y)
            # self.train_data.append([y, self.flipped])
        # else:
        #     self.defend(units=self.attack_group)


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, MarineOverflow()),
        Computer(Race.Zerg, Difficulty.Medium)
    ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Terran, JarexTerran())
    # ], realtime=False)