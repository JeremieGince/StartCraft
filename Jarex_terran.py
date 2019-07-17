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


class JarexTerran(sc2.BotAI):
    RACE = Race.Terran

    MAX_WORKERS_COUNT = 50
    AVG_WORKERS_PER_CMDC = 16

    MAX_BARRACKS_COUNT = 7
    AVG_BARRACK_PER_CMDC = 2
    AVG_DEFENDER_PER_CMDC = 10

    MIN_SUPPLY_LEFT = 10
    MAX_SUPPLY = 200

    attack_group = list()
    defend_group = list()

    MILITARY_UNIT_CLASS = {MARINE: {"max": 50, "priority": 1, "maker_class": BARRACKS, "supply": 1},
                           SIEGETANK: {"max": 0, "priority": 20, "maker_class": FACTORY, "supply": 4},
                           UnitTypeId.THOR: {"max": 10, "priority": 1, "maker_class": FACTORY, "supply": 6},
                           UnitTypeId.CYCLONE: {"max": 10, "priority": 1, "maker_class": FACTORY, "supply": 2},
                           VIKING: {"max": 0, "priority": 50, "maker_class": STARPORT, "supply": 2},
                           UnitTypeId.BATTLECRUISER: {"max": 10, "priority": 1, "maker_class": STARPORT, "supply": 6},
                           UnitTypeId.MARAUDER: {"max": 0, "priority": 3, "maker_class": BARRACKS, "supply": 2},
                           UnitTypeId.GHOST: {"max": 15, "priority": 2, "maker_class": BARRACKS, "supply": 2},
                           UnitTypeId.MEDIVAC: {"max": 2, "priority": 50, "maker_class": STARPORT, "supply": 6}}
    CIVIL_UNIT_CLASS = {UnitTypeId.SCV: {"max": 50, "priority": 1,
                                         "maker_class": UnitTypeId.COMMANDCENTER, "supply": 1}}

    BUILDING_CLASS = {COMMANDCENTER,
                      REFINERY,
                      SUPPLYDEPOT,
                      BARRACKS,
                      FACTORY,
                      STARPORT,
                      UnitTypeId.ARMORY}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.BARRACKS: {"priority": 1, "max": 5,
                                                      "avg_per_cmdc": 2, "add_on": [UnitTypeId.BARRACKSTECHLAB],
                                                      "upgrade": []},
                                UnitTypeId.FACTORY: {"priority": 10, "max": 5,
                                                     "avg_per_cmdc": 1, "add_on": [UnitTypeId.FACTORYTECHLAB],
                                                     "upgrade": []},
                                UnitTypeId.STARPORT: {"priority": 50, "max": 3,
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
                                                             "upgrade": [UpgradeId.COMBATSHIELD]}
                                }

    army_units = list()
    buildings = list()

    RATIO_DEF_ATT_UNITS = 0.5
    MIN_ARMY_SIZE_FOR_ATTACK = 50

    iteration = 0
    current_actions = []
    expend_count = 0

    phase = {"defend": True,
             "attack": False}

    def __init__(self):
        super(JarexTerran, self).__init__()

    def on_start(self):
        self.army_units = Units(list(), self._game_data)
        self.attack_group = Units(list(), self._game_data)
        self.defend_group = Units(list(), self._game_data)
        self.buildings = Units(list(), self._game_data)

    async def on_unit_created(self, unit):
        if unit.type_id in self.MILITARY_UNIT_CLASS:
            self.army_units.append(unit)
            if random.random() <= self.RATIO_DEF_ATT_UNITS:
                self.defend_group.append(unit)
            else:
                self.attack_group.append(unit)
        elif unit.type_id in self.BUILDING_CLASS:
            self.buildings.append(unit)

    async def on_unit_destroyed(self, unit_tag):
        self.army_units.remove(unit_tag) if unit_tag in self.army_units else 0
        # self.buildings.remove(unit_tag) if unit_tag in self.buildings else 0

    async def on_step(self, iteration):
        self.iteration = iteration

        self.buildings.clear()
        b_groups = [self.units(c).ready for c in self.BUILDING_CLASS]
        for g in b_groups:
            for b in g:
                self.buildings.append(b)
        self.buildings = Units(self.buildings, self._game_data)

        self.defend(units=self.defend_group)
        try:
            await self.distribute_workers()
        except ValueError:
            pass
        self.create_refinery()
        await self.create_supply_depot()
        self.create_workers()
        await self.create_military_buildings()
        await self.create_military_units()

        await self.expand()

        self.attack()

        self.defend_until_die()

        self.redistribute_army()

        await self.execute_actions()

    async def execute_actions(self):
        await self.do_actions(self.current_actions)
        self.current_actions.clear()

    def create_workers(self):
        if len(self.units(COMMANDCENTER)) * self.AVG_WORKERS_PER_CMDC > self.workers.amount < self.MAX_WORKERS_COUNT:
            for cmdc in self.units(COMMANDCENTER).ready.noqueue:
                if self.can_afford(SCV) and self.supply_left > 0:
                    self.current_actions.append(cmdc.train(SCV))

    async def create_supply_depot(self):
        cmdcs = self.units(COMMANDCENTER)
        try:
            if self.supply_cap < self.MAX_SUPPLY:
                if self.supply_left < self.MIN_SUPPLY_LEFT and self.can_afford(SUPPLYDEPOT) \
                        and len(cmdcs) and not self.already_pending(SUPPLYDEPOT):
                    await self.build(SUPPLYDEPOT, near=random.choice(cmdcs).position, max_distance=10)
        except ValueError:
            pass

    def create_refinery(self):
        for cmdc in self.units(COMMANDCENTER).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, cmdc)
            for vaspene in vaspenes:
                if self.can_afford(REFINERY) and not self.units(REFINERY).closer_than(1.0, vaspene).exists \
                        and not self.already_pending(REFINERY):
                    worker = self.select_build_worker(vaspene.position, force=True)
                    if worker is None:
                        break
                    self.current_actions.append(worker.build(REFINERY, vaspene))

    async def create_military_buildings(self):
        cmdcs = self.units(COMMANDCENTER)
        if not cmdcs.amount:
            return None

        for b_class, info in self.MILITARY_BUILDINGS_CLASS.items():
            if not self.iteration % info["priority"]:
                builds = self.units(b_class).ready
                if cmdcs.amount * info["avg_per_cmdc"] > builds.amount \
                        < info["max"] and self.can_afford(b_class) and not self.already_pending(b_class):
                    await self.build(b_class, near=cmdcs.random.position,
                                     max_distance=60, placement_step=(10 if info["add_on"] else 2))
                for b in builds.noqueue:
                    if b.add_on_tag == 0 and info["add_on"]:
                        add_on_choice = random.choice(info["add_on"])
                        if not self.already_pending(add_on_choice):
                            try:
                                await self.do(b.build(add_on_choice))
                            except Exception as e:
                                print("add_on err", e)
                                raise Exception("coliss ", str(e))
                for upgrade in info["upgrade"]:
                    if self.can_afford(upgrade) and builds.noqueue:
                        try:
                            await self.do(builds.noqueue.random.research(upgrade))
                        except AttributeError as e:
                            print("err: ", e, upgrade)
                        # self.current_actions.append()

    async def create_military_units(self):
        for unit_class, info in self.MILITARY_UNIT_CLASS.items():
            if not self.iteration % info["priority"]:
                units = self.units(unit_class)
                makers = self.units(info["maker_class"]).ready.noqueue
                if not makers:
                    continue
                if units.amount < info["max"] and self.supply_left >= info["supply"]:
                    for maker in makers:
                        if self.can_afford(unit_class):
                            self.current_actions.append(maker.train(unit_class))

    async def expand(self):
        try:
            if ((len(self.army_units) * self.AVG_DEFENDER_PER_CMDC > self.AVG_DEFENDER_PER_CMDC * self.units(
                    COMMANDCENTER).amount \
                 and self.units(BARRACKS).amount >= self.AVG_BARRACK_PER_CMDC * self.units(COMMANDCENTER).amount) \
                or len(self.units(SCV).idle) >= self.AVG_WORKERS_PER_CMDC) \
                    and self.can_afford(COMMANDCENTER) and not self.already_pending(COMMANDCENTER):
                await self.expand_now()
                self.expend_count += 1
            elif self.expend_count > self.units(COMMANDCENTER).amount and self.can_afford(COMMANDCENTER) \
                and not self.already_pending(COMMANDCENTER):
                await self.expand_now()
        except ValueError:
            return None

    def redistribute_army(self):
        if len(self.army_units) \
                and len(self.defend_group) > self.RATIO_DEF_ATT_UNITS * len(self.army_units) \
                and ((1 - self.RATIO_DEF_ATT_UNITS) * len(self.army_units)) / 2 > len(self.attack_group):
            redistributed_group = self.defend_group.random_group_of(
                int((1 - self.RATIO_DEF_ATT_UNITS) * len(self.army_units)))
            for unit in redistributed_group:
                self.defend_group.remove(unit)
                self.attack_group.append(unit)

    def defend(self, units):
        if len(self.known_enemy_units):
            for unit in units.idle:
                furthest_building = self.buildings.furthest_to(self.start_location)
                dist = furthest_building.distance_to(self.start_location)
                closest_enemies = self.known_enemy_units.closer_than(dist, unit)
                if closest_enemies:
                    self.current_actions.append(unit.attack(closest_enemies.closest_to(unit)))
                else:
                    self.current_actions.append(unit.move(furthest_building))

        elif len(self.buildings):
            for unit in units.idle:
                # random.shuffle(list(buildings))
                # self.current_actions.extend([unit.move(b.position) for b in buildings])
                self.current_actions.append(unit.move(random.choice(self.buildings).position))

    def defend_until_die(self):
        if len(self.units(COMMANDCENTER)) == 0 and self.known_enemy_units:
            for unit in self.army_units.idle:
                self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))
            for unit in self.workers:
                self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))

    def find_target(self, unit, state):
        if len(self.known_enemy_units) > 0:
            return self.known_enemy_units.closest_to(unit)
        elif len(self.known_enemy_structures) > 0:
            return self.known_enemy_structures.closest_to(unit)
        else:
            return random.choice(self.enemy_start_locations)

    def attack(self):
        if len(self.attack_group) >= self.MIN_ARMY_SIZE_FOR_ATTACK or self.supply_used >= self.MAX_SUPPLY - 10:
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(self.find_target(unit, self.state)))
        elif len(self.known_enemy_units) > 0:
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))
        else:
            if self.attack_group.amount and self.attack_group.random.distance_to(self.attack_group.center) > 50:
                for unit in self.attack_group.idle:
                    self.current_actions.append(unit.move(self.attack_group.center))


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Computer(Race.Zerg, Difficulty.Medium)
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Terran, JarexTerran())
    # ], realtime=False)
