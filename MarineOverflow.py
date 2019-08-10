from JarexTerran import JarexTerran

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
    BOTNAME = "MarineOverflow"
    MIN_ARMY_SIZE_FOR_ATTACK = 25
    RATIO_DEF_ATT_UNITS = 0.05

    MILITARY_UNIT_CLASS = {UnitTypeId.MARINE: {"max": 118, "priority": 1, "maker_class": UnitTypeId.BARRACKS, "supply": 1},
                           UnitTypeId.MEDIVAC: {"max": 3, "priority": 2, "maker_class": UnitTypeId.STARPORT, "supply": 6}}

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
                                                             "upgrade": [UpgradeId.STIMPACK]},
                                UnitTypeId.STARPORT: {"priority": 1, "max": 1,
                                                      "avg_per_cmdc": 1, "add_on": [UnitTypeId.STARPORTTECHLAB],
                                                      "upgrade": []}
                                }

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True):
        super(MarineOverflow, self).__init__(use_model, human_control, debug, take_training_data)
        # faire le scout group, faire un trucs que c'est moi qui fait les choix pendant la game ou random ou par models.
        # faire l'affichage de données, finir d'écouter les tuto pour avoir plus d'idées

    def on_end(self, game_result):
        JarexTerran.on_end(self, game_result)
        # print("Ennemy killed: ", self._game_info.killed_enemy)


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(MarineOverflow.BOTRACE, MarineOverflow(human_control=False, debug=True)),
        Computer(Race.Zerg, Difficulty.Hard)
    ], realtime=True)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(MarineOverflow.BOTRACE, MarineOverflow()),
    #     Bot(JarexTerran.BOTRACE, JarexTerran(False, False))
    # ], realtime=False)
