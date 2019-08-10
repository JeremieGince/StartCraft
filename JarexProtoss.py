import random

import sc2
from sc2 import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.player import Bot, Computer, Human
from sc2.helpers import ControlGroup
from sc2.units import Units
from sc2 import position
import cv2
import asyncio
from JarexSc2 import JarexSc2


class JarexProtoss(JarexSc2):
    BOTRACE = Race.Protoss
    BOTNAME = "JarexProtoss"

    MILITARY_UNIT_CLASS = {UnitTypeId.ZEALOT: {"max": 50, "priority": 5, "maker_class": UnitTypeId.GATEWAY, "supply": 2},  # 6
                           UnitTypeId.STALKER: {"max": 50, "priority": 20, "maker_class": UnitTypeId.GATEWAY, "supply": 2},  # 11
                           UnitTypeId.ADEPT: {"max": 50, "priority": 5, "maker_class": UnitTypeId.GATEWAY, "supply": 2},  # 6
                           UnitTypeId.VOIDRAY: {"max": 50, "priority": 25, "maker_class": UnitTypeId.STARGATE, "supply": 4},  # 13
                           UnitTypeId.COLOSSUS: {"max": 50, "priority": 5, "maker_class": UnitTypeId.ROBOTICSFACILITY,  # 4
                                                 "supply": 6},
                           UnitTypeId.CARRIER: {"max": 50, "priority": 1, "maker_class": UnitTypeId.STARGATE,  # 1
                                                "supply": 6},
                           UnitTypeId.MOTHERSHIP: {"max": 1, "priority": 1, "maker_class": UnitTypeId.NEXUS, "supply": 8}}

    SCOUT_CLASS = {UnitTypeId.OBSERVER: {"max": 1, "priority": 1, "maker_class": UnitTypeId.ROBOTICSFACILITY, "supply": 1}}

    MEDIC_CLASS = {}

    CIVIL_UNIT_CLASS = {UnitTypeId.PROBE: {"max": 70, "priority": 1,
                                           "maker_class": UnitTypeId.NEXUS, "supply": 1}}

    CIVIL_BUILDING_CLASS = {UnitTypeId.NEXUS: {"type": "main"},
                            UnitTypeId.ASSIMILATOR: {"type": "vaspene"},
                            UnitTypeId.PYLON: {"type": "supply"}}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.GATEWAY: {"priority": 1, "max": 5,
                                                     "avg_per_cmdc": 2, "add_on": [],
                                                     "upgrade": []},
                                UnitTypeId.CYBERNETICSCORE: {"priority": 1, "max": 1,
                                                             "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.ROBOTICSFACILITY: {"priority": 1, "max": 1,
                                                              "avg_per_cmdc": 1, "add_on": [],
                                                              "upgrade": []},
                                UnitTypeId.STARGATE: {"priority": 1, "max": 3,
                                                      "avg_per_cmdc": 1, "add_on": [],
                                                      "upgrade": []},
                                UnitTypeId.FORGE: {"priority": 1, "max": 1,
                                                   "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.ROBOTICSBAY: {"priority": 1, "max": 1,
                                                         "avg_per_cmdc": 1, "add_on": [], "upgrade": []},
                                UnitTypeId.FLEETBEACON: {"priority": 1, "max": 1,
                                                         "avg_per_cmdc": 1, "add_on": [], "upgrade": []}
                                }

    RATIO_DEF_ATT_UNITS = 0.005
    MIN_ARMY_SIZE_FOR_ATTACK = 25

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True):
        super(JarexProtoss, self).__init__(use_model, human_control, debug, take_training_data)

        self.MAX_SUPPLY_LEFT = 20

    async def create_military_buildings(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        supply_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")

        cmdcs = self.units(cmdc_type)
        supps = self.units(supply_type).ready
        if cmdcs.amount <= 0 or supps.amount <= 0:
            return None

        for b_class, info in self.MILITARY_BUILDINGS_CLASS.items():
            if not self.iteration % info["priority"]:
                builds = self.units(b_class).ready
                if cmdcs.amount * info["avg_per_cmdc"] > builds.amount \
                        < info["max"] and self.can_afford(b_class) and not self.already_pending(b_class):
                    await self.build(b_class, near=self.random_location_variance(supps.random.position, variance=10),
                                     max_distance=30, placement_step=5)
                for upgrade in info["upgrade"]:
                    if self.can_afford(upgrade) and builds.noqueue:
                        build = builds.noqueue.random
                        if build:
                            print(build, upgrade)
                            try:
                                await self.do(build.research(upgrade))
                                await asyncio.sleep(1e-6)
                                info["upgrade"].remove(upgrade)
                            except Exception:
                                pass

    async def create_supply_depot(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        supply_depot_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")
        cmdcs = self.units(cmdc_type)
        try:
            if self.supply_cap < self.MAX_SUPPLY:
                if self.supply_left < self.MAX_SUPPLY_LEFT and self.can_afford(supply_depot_type) \
                        and cmdcs.amount and not self.already_pending(supply_depot_type):
                    pos = self.random_location_variance(cmdcs.random.position.towards(self.game_info.map_center, 5), 10)
                    await self.build(supply_depot_type, near=pos, max_distance=10)
        except ValueError:
            pass


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(JarexProtoss.BOTRACE, JarexProtoss(human_control=False, debug=True)),
        Computer(Race.Terran, Difficulty.Medium)
    ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Terran, JarexTerran())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Human(Race.Zerg, name="Dummy"),
    #     Bot(JarexProtoss.BOTRACE, JarexProtoss(human_control=False, debug=False), name=JarexProtoss.BOTNAME)
    # ], realtime=True)
