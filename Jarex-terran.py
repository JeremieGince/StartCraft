import random

import sc2
from sc2 import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer
from sc2.helpers import ControlGroup

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
    MAX_WORKERS_COUNT = 65
    AVG_WORKERS_PER_CMDC = 16

    MAX_BARRACKS_COUNT = 7
    AVG_BARRACK_PER_CMDC = 1
    AVG_DEFENDER_PER_CMDC = 25

    MIN_SUPPLY_LEFT = 10
    MAX_SUPPLY = 200

    attack_group = list()
    defend_group = list()

    MILITARY_UNIT_CLASS = {MARINE: 150,
                           SIEGETANK: 20,
                           VIKING: 20,
                           UnitTypeId.BATTLECRUISER: 10,
                           UnitTypeId.MARAUDER: 25,
                           UnitTypeId.GHOST: 25}

    BUILDING_CLASS = [COMMANDCENTER, REFINERY, SUPPLYDEPOT, BARRACKS, FACTORY, STARPORT, UnitTypeId.ARMORY]

    army_units = list()
    buildings = list()

    RATIO_DEF_ATT_UNITS = 0.5
    MIN_ARMY_SIZE_FOR_ATTACK = 75

    iteration = 0
    current_actions = []
    expend_count = 0

    phase = {"defend": True,
             "attack": False}

    def __init__(self):
        super(JarexTerran, self).__init__()

    def on_start(self):
        pass

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
        self.buildings.remove(unit_tag) if unit_tag in self.buildings else 0

    async def on_step(self, iteration):
        self.iteration = iteration

        self.defend()
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
        try:
            if self.supply_used < self.MAX_SUPPLY:
                cmdcs = self.units(COMMANDCENTER)
                if self.supply_left < self.MIN_SUPPLY_LEFT and self.can_afford(SUPPLYDEPOT) \
                        and cmdcs.amount:
                    await self.build(SUPPLYDEPOT, near=random.choice(cmdcs).position)
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
        if cmdcs.amount * self.AVG_BARRACK_PER_CMDC > self.units(BARRACKS).amount \
                < self.MAX_BARRACKS_COUNT \
                and self.can_afford(BARRACKS) \
                and not self.already_pending(BARRACKS):
            await self.build(BARRACKS, near=random.choice(cmdcs).position)
        else:
            for barrack in self.units(BARRACKS).ready.noqueue:
                abilities = await self.get_available_abilities(barrack)
                if AbilityId.BUILD_TECHLAB_FACTORY in abilities and \
                        self.can_afford(AbilityId.BUILD_TECHLAB_FACTORY):
                    await self.do(barrack(AbilityId.BUILD_TECHLAB_FACTORY))

        if self.units(FACTORY).amount < cmdcs.amount and self.can_afford(FACTORY) \
                and not self.already_pending(FACTORY):
            await self.build(FACTORY, near=random.choice(cmdcs).position)
        else:
            for factory in self.units(FACTORY).ready.noqueue:
                abilities = await self.get_available_abilities(factory)
                if AbilityId.BUILD_TECHLAB_FACTORY in abilities and \
                        self.can_afford(AbilityId.BUILD_TECHLAB_FACTORY):
                    await self.do(factory(AbilityId.BUILD_TECHLAB_FACTORY))

        if self.units(STARPORT).amount < cmdcs.amount and self.can_afford(STARPORT) \
                and not self.already_pending(STARPORT):
            await self.build(STARPORT, near=random.choice(cmdcs).position)

        if not self.units(UnitTypeId.ARMORY).amount and self.can_afford(UnitTypeId.ARMORY) \
                and not self.already_pending(UnitTypeId.ARMORY):
            await self.build(UnitTypeId.ARMORY, near=random.choice(cmdcs))

        if not self.units(UnitTypeId.FUSIONCORE).amount and self.can_afford(UnitTypeId.FUSIONCORE) \
                and not self.already_pending(UnitTypeId.FUSIONCORE):
            await self.build(UnitTypeId.FUSIONCORE, near=random.choice(cmdcs))

    async def create_military_units(self):
        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(UnitTypeId.GHOST) and self.supply_left >= 2 \
                    and self.units(UnitTypeId.GHOST).amount < self.MILITARY_UNIT_CLASS[UnitTypeId.GHOST]:
                self.current_actions.append(barrack.train(UnitTypeId.GHOST))
            elif self.can_afford(MARINE) and self.supply_left >= 1 \
                    and self.units(MARINE).amount < self.MILITARY_UNIT_CLASS[MARINE]:
                self.current_actions.append(barrack.train(MARINE))
            if self.can_afford(UnitTypeId.MARAUDER) and self.supply_left >= 2 \
                    and self.units(UnitTypeId.MARAUDER).amount < self.MILITARY_UNIT_CLASS[UnitTypeId.MARAUDER]:
                self.current_actions.append(barrack.train(UnitTypeId.MARAUDER))

        for factory in self.units(FACTORY).ready.noqueue:
            if self.can_afford(SIEGETANK) and self.supply_left >= 4 \
                    and self.units(SIEGETANK).amount < self.MILITARY_UNIT_CLASS[SIEGETANK]:
                self.current_actions.append((factory.train(SIEGETANK)))

        for starport in self.units(STARPORT).ready.noqueue:
            if self.can_afford(UnitTypeId.BATTLECRUISER) and self.supply_left >= 6 \
                    and self.units(UnitTypeId.BATTLECRUISER).amount \
                    < self.MILITARY_UNIT_CLASS[UnitTypeId.BATTLECRUISER]:
                self.current_actions.append(starport.train(UnitTypeId.BATTLECRUISER))
            elif self.can_afford(VIKING) and self.supply_left >= 2 \
                    and self.units(VIKING).amount < self.MILITARY_UNIT_CLASS[VIKING]:
                self.current_actions.append(starport.train(VIKING))

    async def expand(self):
        if len(self.army_units) * self.AVG_DEFENDER_PER_CMDC > self.AVG_DEFENDER_PER_CMDC * self.units(
                COMMANDCENTER).amount \
                and self.units(BARRACKS).amount >= self.AVG_BARRACK_PER_CMDC * self.units(COMMANDCENTER).amount \
                and self.can_afford(COMMANDCENTER):
            await self.expand_now()
            self.expend_count += 1

    def defend(self):
        buildings = self.units(COMMANDCENTER) | self.units(BARRACKS) | self.units(FACTORY) | self.units(STARPORT)
        if len(self.known_enemy_units):
            self.current_actions.extend(
                [unit.attack(random.choice(self.known_enemy_units)) for unit in self.defend_group])
        elif len(buildings):
            for unit in self.defend_group:
                # random.shuffle(list(buildings))
                # self.current_actions.extend([unit.move(b.position) for b in buildings])
                self.current_actions.append(unit.move(random.choice(buildings).position))

    def defend_until_die(self):
        if len(self.units(COMMANDCENTER)) == 0 and self.known_enemy_units:
            for unit in self.army_units:
                self.current_actions.append(unit.attack(random.choice(self.known_enemy_units)))
            for unit in self.workers:
                self.current_actions.append(unit.attack(random.choice(self.known_enemy_units)))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return random.choice(self.enemy_start_locations)

    def attack(self):
        if len(self.attack_group) >= self.MIN_ARMY_SIZE_FOR_ATTACK or self.supply_used == self.MAX_SUPPLY:
            for unit in self.attack_group:
                self.current_actions.append(unit.attack(self.find_target(self.state)))
        elif len(self.known_enemy_units) > 0:
            for unit in self.attack_group:
                self.current_actions.append(unit.attack(random.choice(self.known_enemy_units)))


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, JarexTerran()),
        Computer(Race.Zerg, Difficulty.Medium)
    ], realtime=False)
