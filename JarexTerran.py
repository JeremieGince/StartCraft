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
from sc2 import position
import cv2
from JarexSc2 import JarexSc2

COMMANDCENTER = UnitTypeId.COMMANDCENTER
MARINE = UnitTypeId.MARINE
SCV = UnitTypeId.SCV
BARRACKS = UnitTypeId.BARRACKS
SUPPLYDEPOT = UnitTypeId.SUPPLYDEPOT
REFINERY = UnitTypeId.REFINERY
FACTORY = UnitTypeId.FACTORY
SIEGETANK = UnitTypeId.SIEGETANK
STARPORT = UnitTypeId.STARPORT
VIKING = UnitTypeId.VIKINGFIGHTER


class JarexTerran(JarexSc2):
    BOTRACE = Race.Terran
    BOTNAME = "JarexTerran"

    MILITARY_UNIT_CLASS = {MARINE: {"max": 50, "priority": 25, "maker_class": BARRACKS, "supply": 1},
                           SIEGETANK: {"max": 50, "priority": 1, "maker_class": FACTORY, "supply": 4},
                           UnitTypeId.THOR: {"max": 50, "priority": 1, "maker_class": FACTORY, "supply": 6},
                           UnitTypeId.CYCLONE: {"max": 50, "priority": 1, "maker_class": FACTORY, "supply": 2},
                           VIKING: {"max": 50, "priority": 1, "maker_class": STARPORT, "supply": 2},
                           UnitTypeId.BATTLECRUISER: {"max": 50, "priority": 1, "maker_class": STARPORT, "supply": 6},
                           UnitTypeId.MARAUDER: {"max": 50, "priority": 3, "maker_class": BARRACKS, "supply": 2},
                           UnitTypeId.GHOST: {"max": 50, "priority": 10, "maker_class": BARRACKS, "supply": 2},
                           UnitTypeId.MEDIVAC: {"max": 50, "priority": 10, "maker_class": STARPORT, "supply": 6}}

    SCOUT_CLASS = {UnitTypeId.RAVEN: {"max": 1, "priority": 1, "maker_class": STARPORT, "supply": 1},
                   MARINE: {"max": 1, "priority": 1, "maker_class": BARRACKS, "supply": 1}}

    MEDIC_CLASS = {UnitTypeId.MEDIVAC: {"max": 2, "priority": 50, "maker_class": STARPORT, "supply": 6}}

    CIVIL_UNIT_CLASS = {UnitTypeId.SCV: {"max": 64, "priority": 1,
                                         "maker_class": UnitTypeId.COMMANDCENTER, "supply": 1}}

    CIVIL_BUILDING_CLASS = {COMMANDCENTER: {"type": "main"},
                            REFINERY: {"type": "vaspene"},
                            SUPPLYDEPOT: {"type": "supply"}}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.BARRACKS: {"priority": 1, "max": 5,
                                                      "avg_per_cmdc": 2, "add_on": [UnitTypeId.BARRACKSTECHLAB],
                                                      "upgrade": []},
                                UnitTypeId.FACTORY: {"priority": 10, "max": 5,
                                                     "avg_per_cmdc": 1, "add_on": [UnitTypeId.FACTORYTECHLAB],
                                                     "upgrade": []},
                                UnitTypeId.STARPORT: {"priority": 1, "max": 3,
                                                      "avg_per_cmdc": 1, "add_on": [UnitTypeId.STARPORTTECHLAB],
                                                      "upgrade": []},
                                UnitTypeId.ARMORY: {"priority": 1, "max": 1,
                                                    "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.FUSIONCORE: {"priority": 1, "max": 1,
                                                        "avg_per_cmdc": 1, "add_on": [],
                                                        "upgrade": [UpgradeId.YAMATOCANNON]},
                                UnitTypeId.MISSILETURRET: {"priority": 100, "max": 25,
                                                           "avg_per_cmdc": 3, "add_on": [], "upgrade": []},

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
                                                             "upgrade": []}
                                }

    RATIO_DEF_ATT_UNITS = 0.05
    MIN_ARMY_SIZE_FOR_ATTACK = 50

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True):
        super(JarexTerran, self).__init__(use_model, human_control, debug, take_training_data)

    async def create_supply_depot(self):
        await super(JarexTerran, self).create_supply_depot()
        for supply_dep in self.units(UnitTypeId.SUPPLYDEPOT).ready:
            supply_dep(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

    async def create_military_buildings(self):
        await super(JarexTerran, self).create_military_buildings()

        barracks = self.units(UnitTypeId.BARRACKS).ready.noqueue
        if barracks:
            barrack = barracks.random
            if barrack.add_on_tag == 0 and self.can_afford(UnitTypeId.BARRACKSTECHLAB) \
                    and not self.already_pending(UnitTypeId.BARRACKSTECHLAB) \
                    and not self.units(UnitTypeId.BARRACKSTECHLAB).amount:
                try:
                    await self.do(barrack.build(UnitTypeId.BARRACKSTECHLAB))
                except Exception:
                    return None


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, JarexTerran()),
        Computer(Race.Zerg, Difficulty.Hard)
    ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Terran, JarexTerran())
    # ], realtime=False)
