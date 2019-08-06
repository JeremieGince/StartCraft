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
import numpy as np
import math
import asyncio
import os
import time

from getkeys import key_check


class JarexSc2(sc2.BotAI):
    BOTRACE = None
    BOTNAME = "JarexSc2"

    AVG_DEFENDER_PER_CMDC = 10

    MAX_SUPPLY_LEFT = 10
    MAX_SUPPLY = 200

    MILITARY_UNIT_CLASS = dict()

    SCOUT_CLASS = dict()

    MEDIC_CLASS = dict()

    CIVIL_UNIT_CLASS = dict()

    CIVIL_BUILDING_CLASS = dict()

    MILITARY_BUILDINGS_CLASS = dict()

    army_units = list()
    attack_group = list()
    defend_group = list()
    scout_group = list()
    medic_group = list()
    buildings = list()

    scout_points = list()
    hm_scout_per_ennemy = 3

    RATIO_DEF_ATT_UNITS = 0.5
    MIN_ARMY_SIZE_FOR_ATTACK = 50

    iteration = 0
    current_actions = []
    expend_count = 1

    phase = {"defend": True,
             "attack": False}

    do_something_after = 0

    def __init__(self, use_model, human_control=False, debug=False, take_training_data=True):
        super(JarexSc2, self).__init__()
        self.use_model = use_model
        self.human_control = human_control
        self.debug = debug
        self.take_training_data = take_training_data
        assert (use_model and human_control) is False, "Both use_model and human_control, can't be True"

        self.intel_out = None

        self.attack_group_choices = {0: self.defend_attack_group,
                                     1: self.attack_know_enemy_units,
                                     2: self.attack_known_enemy_structures,
                                     3: self.attack_enemy_start_location,
                                     4: self.do_nothing}
        self.current_action_choice = 4
        if self.human_control:
            print(f"attack group choices {self.attack_group_choices}")

        self.training_data = {"params": {}, "data": list()}

    def on_start(self):
        self.army_units = Units(list(), self._game_data)
        self.attack_group = Units(list(), self._game_data)
        self.defend_group = Units(list(), self._game_data)
        self.scout_group = Units(list(), self._game_data)
        self.medic_group = Units(list(), self._game_data)
        self.buildings = Units(list(), self._game_data)

        # self.create_scout_points()

        print(f"we need {self.hm_scout_per_ennemy*len(self.enemy_start_locations)} Scout")
        for unit_type, info in self.SCOUT_CLASS.items():
            info["max"] = self.hm_scout_per_ennemy*len(self.enemy_start_locations)

        self.intel_out = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 1), np.uint8)

    def on_end(self, game_result):
        print("--- on_end called ---")
        print(f"game_result {game_result}")
        print(f"train_data amount: {len(self.training_data['data'])}")
        # print(f"Ennemy killed: {self._game_info.killed_enemy}")

        if game_result == sc2.Result.Victory and self.take_training_data:
            folder = f"train_data_{len(self.attack_group_choices)}_choices"
            filename = f"trdata_{time.strftime('%Y%m%d%H%M%S')}_{len(self.training_data['data'])}.npy"
            if not os.path.exists(f"training_data/{folder}"):
                os.makedirs(f"training_data/{folder}")
            np.save(f"training_data/{folder}/{filename}", self.training_data)

    async def on_unit_created(self, unit):
        if unit.type_id in self.SCOUT_CLASS and not self.scout_group_is_complete():
            self.scout_group.append(unit)
        elif unit.type_id in self.MILITARY_UNIT_CLASS:
            self.army_units.append(unit)
            if random.random() <= self.RATIO_DEF_ATT_UNITS:
                self.defend_group.append(unit)
                print(f"defend_group: {self.defend_group}")
            else:
                self.attack_group.append(unit)
        elif unit.type_id in self.CIVIL_BUILDING_CLASS:
            self.buildings.append(unit)

    async def on_unit_destroyed(self, unit_tag):
        groups = [self.army_units, self.attack_group, self.defend_group,
                  self.scout_group, self.medic_group, self.buildings]
        for group in groups:
            self.delete_unit_tag_from_group(group, unit_tag)

    def delete_unit_tag_from_group(self, group, unit_tag):
        unit = group.tags_in([unit_tag])
        if unit:
            unit = unit[0]
            group.remove(unit)
            # print(f"{unit} died and removed {unit not in group} from {group}")

    def scout_group_is_complete(self):
        return self.scout_group.amount >= self.hm_scout_per_ennemy*len(self.enemy_start_locations)

    async def on_step(self, iteration):
        self.iteration = iteration

        await self.expand()

        self.buildings.clear()
        b_groups = [self.units(c).ready for c in self.CIVIL_BUILDING_CLASS]
        for g in b_groups:
            for b in g:
                self.buildings.append(b)
        self.buildings = Units(self.buildings, self._game_data)
        await self.intel()

        await self.scout()

        self.defend(units=self.defend_group)
        try:
            await self.distribute_workers()
        except ValueError:
            pass

        try:
            self.create_refinery()
        except Exception:
            pass

        await self.create_supply_depot()

        try:
            self.create_workers()
        except Exception:
            pass

        try:
            await self.create_military_buildings()
        except Exception:
            pass

        await self.create_military_units()

        await self.take_action()

        # self.defend_until_die()

        self.redistribute_army()

        await self.execute_actions()

    async def execute_actions(self):
        try:
            await self.do_actions(self.current_actions)
        except Exception:
            pass
        self.current_actions.clear()

    def create_workers(self):
        for worker_type, info in self.CIVIL_UNIT_CLASS.items():
            if self.units(info["maker_class"]).amount:
                for maker in self.units(info["maker_class"]).ready.noqueue:
                    if self.can_afford(worker_type) and self.supply_left > 0 \
                            and maker.ideal_harvesters > maker.assigned_harvesters \
                            and self.workers.amount < info["max"]:
                        self.current_actions.append(maker.train(worker_type))

    async def create_supply_depot(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        supply_depot_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")
        cmdcs = self.units(cmdc_type)
        try:
            if self.supply_cap < self.MAX_SUPPLY:
                if self.supply_left < self.MAX_SUPPLY_LEFT and self.can_afford(supply_depot_type) \
                        and cmdcs.amount and not self.already_pending(supply_depot_type):
                    await self.build(supply_depot_type, near=self.random_location_variance(cmdcs.random.position, 10),
                                     max_distance=10)
        except ValueError:
            pass

    def create_refinery(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        vaspene_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "vaspene")
        for cmdc in self.units(cmdc_type).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, cmdc)
            for vaspene in vaspenes:
                if self.can_afford(vaspene_type) and not self.units(vaspene_type).closer_than(1.0, vaspene).exists \
                        and not self.already_pending(vaspene_type):
                    worker = self.select_build_worker(vaspene.position, force=True)
                    if worker is None:
                        break
                    self.current_actions.append(worker.build(vaspene_type, vaspene))

    async def create_military_buildings(self):
        try:
            cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
            cmdcs = self.units(cmdc_type)
            if not cmdcs.amount:
                return None

            for b_class, info in self.MILITARY_BUILDINGS_CLASS.items():
                if not self.iteration % info["priority"]:
                    builds = self.units(b_class).ready
                    if cmdcs.amount * info["avg_per_cmdc"] > builds.amount \
                            < info["max"] and self.can_afford(b_class) and not self.already_pending(b_class):
                        await self.build(b_class,
                                         near=self.random_location_variance(cmdcs.random.position, variance=10),
                                         max_distance=100, placement_step=(30 if info["add_on"] else 10))
                    for b in builds.noqueue:
                        if b.add_on_tag == 0 and info["add_on"]:
                            add_on_choice = random.choice(info["add_on"])
                            if not self.already_pending(add_on_choice):
                                try:
                                    await self.do(b.build(add_on_choice))
                                except Exception as e:
                                    print("add_on err", e)
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
        except Exception:
            pass

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

    async def build_scout(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        cmdcs = self.units(cmdc_type)
        if not cmdcs.amount:
            return None

        nb_to_build = len(self.enemy_start_locations) - self.scout_group.amount
        for _ in range(nb_to_build):
            random_scout_class = random.choice(list(self.SCOUT_CLASS.keys()))
            random_scout_info = self.SCOUT_CLASS[random_scout_class]
            makers = self.units(random_scout_info["maker_class"])
            if not makers and self.can_afford(random_scout_info["maker_class"]) \
                and not self.already_pending(random_scout_info["maker_class"]):
                await self.build(random_scout_info["maker_class"], near=cmdcs.random)
                return None
            elif not makers:
                return None
            makers = makers.ready.noqueue
            if makers and self.can_afford(random_scout_class) and not self.already_pending(random_scout_class) \
                    and self.supply_left >= random_scout_info["supply"]:
                self.current_actions.append(makers.random.train(random_scout_class))

    async def expand(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        a = np.sum([c.surplus_harvesters for c in self.units(cmdc_type).ready])
        try:
            if a > 0 and self.can_afford(cmdc_type) and not self.already_pending(cmdc_type):
                await self.expand_now()
                self.expend_count += 1
            elif self.expend_count > self.units(cmdc_type).amount and self.can_afford(cmdc_type) \
                    and not self.already_pending(cmdc_type):
                # print(f"We lost {self.expend_count-self.units(cmdc_type).amount} cmdc")
                await self.expand_now()
            elif self.expend_count == 1 and self.can_afford(cmdc_type) and not self.already_pending(cmdc_type):
                await self.expand_now()
                self.expend_count += 1
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
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        if len(self.units(cmdc_type)) == 0 and self.known_enemy_units:
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

    async def defend_attack_group(self):
        self.defend(units=self.attack_group)

    async def attack_know_enemy_units(self):
        if len(self.known_enemy_units):
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))

    async def attack_known_enemy_structures(self):
        if len(self.known_enemy_structures):
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(self.known_enemy_structures.closest_to(unit)))

    async def attack_enemy_start_location(self):
        if self.enemy_start_locations:
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(random.choice(self.enemy_start_locations)))

    async def do_nothing(self):
        # wait = random.randrange(20, 165)
        # self.do_something_after = self.iteration + wait
        pass

    async def take_action(self):
        if self.iteration > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.intel_out.reshape([-1, 176, 200, 3])])
                choice = np.argmax(prediction[0])
                # print('prediction: ',choice)

            elif self.human_control:
                choice = key_check([str(c) for c in list(self.attack_group_choices.keys())])
                if not len(choice):
                    print(f"You take no choice")
                    return None
                else:
                    choice = choice[0]
                    print(f"You take {choice}")
            else:

                defend_weight = 1 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 10
                attack_units_weight = 10 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 5
                attack_struct_weight = 4 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 0
                attack_estl_weight = 4 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 0
                do_nothing_weight = 1 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 1

                weights = [defend_weight, attack_units_weight, attack_struct_weight, attack_estl_weight,
                           do_nothing_weight]
                choices = list(self.attack_group_choices.keys())
                # weighted_choices = sum([weights[i] * [choices[i]] for i in range(len(choices))])
                assert len(weights) == len(choices)
                weighted_choices = list()
                for i, c in enumerate(choices):
                    for _ in range(weights[i]):
                        weighted_choices.append(c)
                choice = random.choice(weighted_choices)

            self.current_action_choice = choice
            wait = random.randrange(25, 85)
            self.do_something_after = self.iteration + wait

            y = np.zeros(len(self.attack_group_choices))
            y[choice] = 1
            print(y)
            self.training_data["data"].append([y, self.intel_out])
        try:
            await self.attack_group_choices[self.current_action_choice]()
        except Exception as e:
            print(str(e))

    def random_location_variance(self, location, variance=100):
        x = location[0]
        y = location[1]

        x += random.randrange(-variance, variance)
        y += random.randrange(-variance, variance)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y)))

        return go_to

    async def scout(self):
        '''
        ['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_game_data', '_proto', '_type_data', 'add_on_tag', 'alliance', 'assigned_harvesters', 'attack', 'build', 'build_progress', 'cloak', 'detect_range', 'distance_to', 'energy', 'facing', 'gather', 'has_add_on', 'has_buff', 'health', 'health_max', 'hold_position', 'ideal_harvesters', 'is_blip', 'is_burrowed', 'is_enemy', 'is_flying', 'is_idle', 'is_mine', 'is_mineral_field', 'is_powered', 'is_ready', 'is_selected', 'is_snapshot', 'is_structure', 'is_vespene_geyser', 'is_visible', 'mineral_contents', 'move', 'name', 'noqueue', 'orders', 'owner_id', 'position', 'radar_range', 'radius', 'return_resource', 'shield', 'shield_max', 'stop', 'tag', 'train', 'type_id', 'vespene_contents', 'warp_in']
        '''

        if self.scout_group.amount > 0:
            self.create_scout_points()
            for i, scout in enumerate(self.scout_group):
                if scout.is_idle:
                    move_to = self.random_location_variance(self.scout_points[i])
                    self.current_actions.append(scout.move(move_to))

        if not self.scout_group_is_complete():
            await self.build_scout()

    def create_scout_points(self):
        for esl in self.enemy_start_locations:
            expand_dis_dir = {}

            for el in self.expansion_locations:
                distance_to_enemy_start = el.distance_to(esl)
                # print(distance_to_enemy_start)
                expand_dis_dir[distance_to_enemy_start] = el

            ordered_exp_distances = sorted(list(expand_dis_dir.keys()))
            self.scout_points.extend([expand_dis_dir[dist] for dist in ordered_exp_distances])

    async def intel3Channels(self):
        raise NotImplementedError("Intel")

        # for game_info: https://github.com/Dentosal/python-sc2/blob/master/sc2/game_info.py#L162
        # print(self.game_info.map_size)
        # flip around. It's y, x when you're dealing with an array.
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # UNIT: [SIZE, (BGR COLOR)]
        '''from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STARGATE, VOIDRAY'''
        draw_dict = {
            NEXUS: [15, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            PROBE: [1, (55, 200, 0)],
            ASSIMILATOR: [2, (55, 200, 0)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            STARGATE: [5, (255, 0, 0)],
            ROBOTICSFACILITY: [5, (215, 155, 0)],
            # VOIDRAY: [3, (255, 100, 0)],
        }

        for unit_type in draw_dict:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        # NOT THE MOST IDEAL, BUT WHATEVER LOL
        main_base_names = ["nexus", "commandcenter", "hatchery"]
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        for obs in self.units(OBSERVER).ready:
            pos = obs.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        for vr in self.units(VOIDRAY).ready:
            pos = vr.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (255, 100, 0), -1)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),
                 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),
                 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to make our final fix in visual representation:
        self.flipped = cv2.flip(game_data, 0)
        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

        if not HEADLESS:
            if self.use_model:
                cv2.imshow('Model Intel', resized)
                cv2.waitKey(1)
            else:
                cv2.imshow('Random Intel', resized)
                cv2.waitKey(1)

    async def intel(self):
        '''
        just simply iterate units.

        outline fighters in white possibly?

        draw pending units with more alpha

        '''

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius), (255, 255, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

        try:
            line_max = 50
            mineral_ratio = self.minerals / 1500
            if mineral_ratio > 1.0:
                mineral_ratio = 1.0

            vespene_ratio = self.vespene / 1500
            if vespene_ratio > 1.0:
                vespene_ratio = 1.0

            population_ratio = self.supply_left / self.supply_cap
            if population_ratio > 1.0:
                population_ratio = 1.0

            plausible_supply = self.supply_cap / self.MAX_SUPPLY

            worker_type = list(self.CIVIL_UNIT_CLASS.keys())[0]
            worker_weight = self.units(worker_type).amount / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            # print(str(e))
            pass
        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.intel_out = cv2.flip(grayed, 0)

        if self.debug:
            resized = cv2.resize(self.intel_out, dsize=None, fx=2, fy=2)

            cv2.imshow(str(self.BOTNAME), resized)
            cv2.waitKey(1)

    @staticmethod
    def find_key_per_info(dic, info_key, info_value):
        re_key = None
        for key, info in dic.items():
            if info[info_key] == info_value:
                re_key = key
                break
        return re_key


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
