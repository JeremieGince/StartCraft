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
from QUnit import QUnit, Unit
from Deep.Models import Sc2UnitMakerNet
import torch


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

    DEFENSE_BUILDING_CLASS = dict()

    RESERVE_BUILDING_CLASS = dict()

    Q_CLASS = dict()

    DAMAGE_UNITS_ABILITIES = dict()
    ATTACK_UNITS_ABILITIES = dict()

    army_units = list()
    attack_group = list()
    defend_group = list()
    scout_group = list()
    medic_group = list()
    q_group = list()
    buildings = list()

    scout_points = list()
    hm_scout_per_ennemy = 3

    RATIO_DEF_ATT_UNITS = 0.5
    MIN_ARMY_SIZE_FOR_ATTACK = 65
    MIN_ARMY_SIZE_FOR_RETRETE = 15

    iteration = 0
    current_actions = []
    expend_count = 1

    phase = {"defend": True,
             "attack": False}

    do_something_after = 0
    do_something_after_scout = 0
    iteration_per_scouting_point = 250

    MIN_EXPENSION = 2

    hm_iteration_per_efficacity_compute = 250

    def __init__(self, use_model, human_control=False, debug=False, take_training_data=True, epsilon=0.05):
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

        self.training_data = {"params": {}, "actions_choice_data": list(), "create_units_data": list()}

        self.DIRECTORY_PATH_ACTION_MAKER = f"train_data_{self.BOTNAME}_{len(self.attack_group_choices)}_choices"
        self.DIRECTORY_PATH_UNIT_MAKER = f"train_data_{self.BOTNAME}_{len(self.MILITARY_UNIT_CLASS) + 1}_units"

        if self.use_model:
            try:
                self.action_model = torch.load(f"Models/{self.BOTNAME}_action_model_Harder.pth").cpu()
            except FileNotFoundError:
                self.action_model = torch.load(f"../Models/{self.BOTNAME}_action_model.pth").cpu()

            self.unit_maker_model = Sc2UnitMakerNet(self.BOTNAME)
            self.unit_maker_model.load()
        else:
            self.action_model = None
            self.unit_maker_model = None
        self.epsilon = epsilon

        self.total_enemy_killed = 0
        self.last_iteration_enemy_killed = 0
        self.killed_score = 0
        self.last_killed_score = 0
        self.is_attacking_ratio = 0.0

    def on_start(self):
        self.army_units = Units(list(), self._game_data)
        self.attack_group = Units(list(), self._game_data)
        self.defend_group = Units(list(), self._game_data)
        self.scout_group = Units(list(), self._game_data)
        self.medic_group = Units(list(), self._game_data)
        self.q_group = Units(list(), self._game_data)
        self.buildings = Units(list(), self._game_data)

        # self.create_scout_points()

        print(f"we need {self.hm_scout_per_ennemy*len(self.enemy_start_locations)} Scout")
        for unit_type, info in self.SCOUT_CLASS.items():
            info["max"] = self.hm_scout_per_ennemy*len(self.enemy_start_locations)

        self.intel_out = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 1), np.uint8)

    def on_end(self, game_result):
        print("--- on_end called ---")
        print(f"game_result {game_result}")
        # print(f"train_data amount: {len(self.training_data['actions_choice_data'])}")
        # print(f"Ennemy killed: {self._game_info.killed_enemy}")

        if game_result == sc2.Result.Victory and self.take_training_data:

            # Action Maker saving Data
            ac_filename = f"trdata_{time.strftime('%Y%m%d%H%M%S')}_{len(self.training_data['actions_choice_data'])}.npy"

            if not os.path.exists(f"training_data/{self.DIRECTORY_PATH_ACTION_MAKER}"):
                os.makedirs(f"training_data/{self.DIRECTORY_PATH_ACTION_MAKER}")

            np.save(f"training_data/{self.DIRECTORY_PATH_ACTION_MAKER}/{ac_filename}",
                    {"data": self.training_data["actions_choice_data"]})

            # Unit Maker saving Data
            um_filename = f"trdata_{time.strftime('%Y%m%d%H%M%S')}_{len(self.training_data['create_units_data'])}.npy"

            if not os.path.exists(f"training_data/{self.DIRECTORY_PATH_UNIT_MAKER}"):
                os.makedirs(f"training_data/{self.DIRECTORY_PATH_UNIT_MAKER}")

            np.save(f"training_data/{self.DIRECTORY_PATH_UNIT_MAKER}/{um_filename}",
                    {"data": self.training_data["create_units_data"]})

    async def on_unit_created(self, unit):
        if unit.type_id in self.SCOUT_CLASS and not self.scout_group_is_complete():
            self.scout_group.append(unit)
        elif unit.type_id in self.Q_CLASS and not self.q_group_is_complete():
            self.q_group.append(QUnit(unit, self.q_group, self))
        elif unit.type_id in self.MILITARY_UNIT_CLASS:
            self.army_units.append(unit)
            if random.random() <= self.RATIO_DEF_ATT_UNITS:
                self.defend_group.append(unit)
                print(f"defend_group: {self.defend_group}")
            else:
                self.attack_group.append(unit)

        if unit.type_id in self.MILITARY_UNIT_CLASS:
            self.MILITARY_UNIT_CLASS[unit.type_id]["created"] += 1
            self.MILITARY_UNIT_CLASS[unit.type_id]["created_batch"] += 1

    async def on_building_construction_complete(self, unit: Unit):
        if unit.type_id in self.CIVIL_BUILDING_CLASS:
            self.buildings.append(unit)

    async def on_unit_destroyed(self, unit_tag):
        groups = [self.army_units, self.attack_group, self.defend_group,
                  self.scout_group, self.medic_group, self.buildings,
                  self.q_group]
        for group in groups:
            self.delete_unit_tag_from_group(group, unit_tag)

    def delete_unit_tag_from_group(self, group, unit_tag):
        unit = group.tags_in([unit_tag])
        if unit:
            unit = unit[0]
            group.remove(unit)
            # print(f"{unit} died and removed {unit not in group} from {group}")
            if unit.type_id in self.MILITARY_UNIT_CLASS:
                self.MILITARY_UNIT_CLASS[unit.type_id]["dead"] += 1
                self.MILITARY_UNIT_CLASS[unit.type_id]["dead_batch"] += 1

    def scout_group_is_complete(self) -> bool:
        return self.scout_group.amount >= self.hm_scout_per_ennemy*len(self.enemy_start_locations)

    def q_group_is_complete(self):
        return self.q_group.amount >= 0

    async def update_q_units(self):
        for unit in self.q_group:
            await unit.take_action()

    def get_is_attacking_ratio(self, units: Units = None):
        if units is None:
            units = self.units.ready
        is_attacking_count = 0
        total = 0
        for u in units:
            try:
                if isinstance(u.is_attacking, bool) and u.is_attacking:
                    is_attacking_count += 1
                total += 1
            except Exception:
                continue
        if total:
            is_attacking_ratio = is_attacking_count/total
        else:
            is_attacking_ratio = 0.0
        return is_attacking_ratio

    async def on_step(self, iteration):
        self.iteration = iteration

        if self.iteration == 1:
            for t in self.townhalls:
                if t not in self.buildings:
                    self.buildings.append(t)

        self.is_attacking_ratio = self.get_is_attacking_ratio()

        await self.expand()

        try:
            await self.scout()
        except Exception:
            pass

        await self.intel()

        self.get_enemy_dead_units()
        self.update_killed_score()

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
            if self.townhalls.amount >= self.MIN_EXPENSION:
                await self.create_military_buildings()
            else:
                await self.build_reserve_buildings()
        except Exception:
            pass

        try:
            await self.build_defense_buildings()
        except Exception:
            pass
        self.compute_units_efficacity()
        # await self.create_military_units_manually()
        await self.create_military_units()
        await self.take_action()
        await self.update_q_units()

        # self.defend_until_die()

        self.redistribute_army()

        await self.put_ability_on_damage_unit()
        await self.put_ability_on_attacking_unit()

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
        supply_depot_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")
        try:
            if self.supply_cap < self.MAX_SUPPLY \
                    and (not self.units(supply_depot_type).amount or self.expend_count > 1):
                if self.supply_left < self.MAX_SUPPLY_LEFT and self.can_afford(supply_depot_type) \
                        and self.townhalls.amount and not self.already_pending(supply_depot_type):
                    await self.build(supply_depot_type,
                                     near=self.random_location_variance(self.townhalls.random.position, 10),
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
            if not self.townhalls.amount and not self.workers.amount:
                return None

            for b_class, info in self.MILITARY_BUILDINGS_CLASS.items():
                pos = self.townhalls.random.position
                worker = self.workers.random
                abilities = await self.get_available_abilities(worker)
                if not self.iteration % info["priority"]:
                    builds = self.units(b_class).ready
                    if self.townhalls.amount * info["avg_per_cmdc"] > builds.amount < info["max"] \
                            and self.can_afford(b_class) and not self.already_pending(b_class) \
                            and info["ability_id"] in abilities:
                        try:
                            await self.build(b_class,
                                             near=self.random_location_variance(pos, variance=10),
                                             max_distance=75, placement_step=(20 if info["add_on"] else 10))
                        except Exception:
                            pass

                    if self.townhalls.amount >= self.MIN_EXPENSION + 1:
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

    async def build_defense_buildings(self):
        try:
            if not self.townhalls.amount and not self.workers.amount:
                return None

            for b_class, info in self.DEFENSE_BUILDING_CLASS.items():
                pos = self.buildings.closest_to(random.choice(self.enemy_start_locations)).position
                worker = self.workers.random
                abilities = await self.get_available_abilities(worker)
                if not self.iteration % info["priority"]:
                    builds = self.units(b_class)
                    if self.townhalls.amount * info["avg_per_cmdc"] > builds.amount < info["max"] \
                        and self.can_afford(b_class) and not self.already_pending(b_class) \
                            and info["ability_id"] in abilities:
                        try:
                            await self.build(b_class,
                                             near=self.random_location_variance(pos, variance=10),
                                             max_distance=30, placement_step=(10 if info["add_on"] else 5))
                        except Exception:
                            continue

        except Exception:
            pass

    async def build_reserve_buildings(self):
        try:
            if not self.townhalls.amount and not self.workers.amount:
                return None

            for b_class, info in self.RESERVE_BUILDING_CLASS.items():
                pos = self.townhalls.closest_to(self.start_location).position
                worker = self.workers.random
                abilities = await self.get_available_abilities(worker)
                if not self.iteration % info["priority"]:
                    builds = self.units(b_class)
                    if self.townhalls.amount * info["avg_per_cmdc"] > builds.amount < info["max"] \
                        and self.can_afford(b_class) and not self.already_pending(b_class) \
                            and info["ability_id"] in abilities:
                        try:
                            await self.build(b_class,
                                             near=self.random_location_variance(pos, variance=10),
                                             max_distance=30, placement_step=(10 if info["add_on"] else 5))
                        except Exception:
                            continue

        except Exception:
            pass

    async def create_military_units_manually(self):
        for unit_class, info in self.MILITARY_UNIT_CLASS.items():
            if not self.iteration % info["priority"]:
                await self.create_unit(unit_class, info)

    async def create_unit(self, unit_class, info=None, n=1):
        check = False

        if info is None:
            info = self.MILITARY_UNIT_CLASS[unit_class]

        units = self.units(unit_class)
        makers = self.units(info["maker_class"]).ready
        if units.amount < info["max"] and self.supply_left >= info["supply"] and makers.amount > 0:
            for _ in range(n):
                makers = makers.noqueue
                if self.can_afford(unit_class) and makers.amount > 0:
                    maker = makers.random
                    abilities = await self.get_available_abilities(maker)
                    try:
                        if info["ability_id"] in abilities:
                            self.current_actions.append(maker.train(unit_class))
                            check = True
                    except Exception:
                        continue
        return check

    async def build_scout(self):

        if not self.townhalls.amount:
            return None

        nb_to_build = self.hm_scout_per_ennemy*len(self.enemy_start_locations) - self.scout_group.amount
        if self.supply_used < self.MAX_SUPPLY:
            for _ in range(nb_to_build):
                random_scout_class = random.choice(list(self.SCOUT_CLASS.keys()))
                random_scout_info = self.SCOUT_CLASS[random_scout_class]
                makers = self.units(random_scout_info["maker_class"])
                if not makers and self.can_afford(random_scout_info["maker_class"]) \
                    and not self.already_pending(random_scout_info["maker_class"]):
                    await self.build(random_scout_info["maker_class"], near=self.townhalls.random.position)
                    return None
                elif not makers:
                    return None
                makers = makers.ready.noqueue
                if makers and self.can_afford(random_scout_class) and not self.already_pending(random_scout_class) \
                        and self.supply_left >= random_scout_info["supply"]:
                    self.current_actions.append(makers.random.train(random_scout_class))
        elif nb_to_build and self.attack_group.amount:
            redistributed_group = self.attack_group.random_group_of(nb_to_build)
            for unit in redistributed_group:
                self.attack_group.remove(unit)
                self.scout_group.append(unit)

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
            elif self.expend_count < self.MIN_EXPENSION and self.can_afford(cmdc_type) and not self.already_pending(cmdc_type):
                await self.expand_now()
                self.expend_count += 1
        except ValueError:
            return None

    def redistribute_army(self):
        try:
            if len(self.army_units) \
                    and len(self.defend_group) > self.RATIO_DEF_ATT_UNITS * len(self.army_units) \
                    and ((1 - self.RATIO_DEF_ATT_UNITS) * len(self.army_units)) / 2 > len(self.attack_group):
                n = int((1 - self.RATIO_DEF_ATT_UNITS) * len(self.army_units))
                if n > 0:
                    redistributed_group = self.defend_group.random_group_of(n)

                    for unit in redistributed_group:
                        self.defend_group.remove(unit)
                        self.attack_group.append(unit)
        except Exception:
            pass

    def defend(self, units):
        if len(self.known_enemy_units):
            for unit in units.idle:
                furthest_building = self.buildings.furthest_to(self.start_location)
                dist = furthest_building.distance_to(self.start_location)
                closest_enemies = self.known_enemy_units.closer_than(dist, unit)
                if closest_enemies:
                    # self.current_actions.append(unit.attack(closest_enemies.closest_to(unit)))
                    self.current_actions.append(unit.attack(closest_enemies.center))
                else:
                    # self.current_actions.append(unit.move(furthest_building))
                    self.current_actions.append(unit.attack(furthest_building.position))

        elif len(self.buildings):
            furthest_building = self.buildings.furthest_to(self.start_location)
            for unit in units.idle:
                # random.shuffle(list(buildings))
                # self.current_actions.extend([unit.move(b.position) for b in buildings])
                # self.current_actions.append(unit.move(random.choice(self.buildings).position))
                # self.current_actions.append(unit.move(furthest_building))
                self.current_actions.append(unit.attack(furthest_building.position))

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
                # self.current_actions.append(unit.attack(self.known_enemy_units.closest_to(unit)))
                self.current_actions.append(unit.attack(self.known_enemy_units.center))

    async def attack_known_enemy_structures(self):
        if len(self.known_enemy_structures):
            for unit in self.attack_group.idle:
                self.current_actions.append(unit.attack(self.known_enemy_structures.closest_to(unit).position))

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
            if self.use_model and random.random() > self.epsilon:
                input_tensor = torch.FloatTensor(self.intel_out[np.newaxis, :, :]).unsqueeze(0).cpu()
                # if torch.cuda.is_available():
                #     input_tensor.cuda()
                prediction = self.action_model(input_tensor).detach().numpy()
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
                if self.attack_group.amount < self.MIN_ARMY_SIZE_FOR_RETRETE \
                        or self.get_is_attacking_ratio(self.buildings) \
                        or (self.current_action_choice == 0 and self.is_attacking_ratio) \
                        or self.get_is_attacking_ratio(self.defend_group):
                    defend_weight = 25
                elif self.attack_group.amount < self.MIN_ARMY_SIZE_FOR_ATTACK:
                    defend_weight = 5
                else:
                    defend_weight = 1

                # defend_weight = 1 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 10
                attack_units_weight = 10 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 2
                attack_struct_weight = 5 if self.attack_group.amount >= self.MIN_ARMY_SIZE_FOR_ATTACK else 0
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
                if self.use_model and self.debug:
                    print("random choice")

            self.current_action_choice = choice
            wait = random.randrange(25, 85)
            self.do_something_after = self.iteration + wait

            y = np.zeros(len(self.attack_group_choices))
            y[choice] = 1
            print(f"action choice: {self.attack_group_choices[choice]}") if self.debug else 0
            self.training_data["actions_choice_data"].append([self.intel_out, y])
        try:
            await self.attack_group_choices[self.current_action_choice]()
        except Exception as e:
            print(str(e))

    async def create_military_units(self):
        if self.use_model and False:
            prediction = self.unit_maker_model(self.get_units_state()[np.newaxis, :])[0]
            choice = int(np.argmax(prediction))

            if choice != len(self.MILITARY_UNIT_CLASS):
                class_choice = list(self.MILITARY_UNIT_CLASS.keys())[choice]
                print(class_choice)
                await self.create_unit(class_choice, n=1)

        elif self.human_control:
            choice = key_check([str(c) for c in list(self.attack_group_choices.keys())])
            if not len(choice):
                print(f"You take no choice")
                return None
            else:
                choice = choice[0]
                print(f"You take {choice}")
        else:
            # weights = [int(i["priority"] * ((i["created"]+1)/(i["dead"]+1))) for _, i in self.MILITARY_UNIT_CLASS.items()]
            # weights = [i["priority"] for _, i in self.MILITARY_UNIT_CLASS.items()]
            weights = list()
            for c, i in self.MILITARY_UNIT_CLASS.items():
                if self.units(c).amount < i["max"] and self.can_afford(c):
                    # i["priority"] = 100
                    if i["type"] == "combat":
                        weights.append(int(round(i["priority"]*i["efficacity"])))
                    else:
                        weights.append(i["priority"])
                else:
                    weights.append(0)
                # print(f"{c} weight: {weights[-1]}") if not self.iteration % 250 else 0

            choices = list(self.MILITARY_UNIT_CLASS.keys())
            # weighted_choices = sum([weights[i] * [choices[i]] for i in range(len(choices))])
            assert len(weights) == len(choices)
            weighted_choices = list()
            for i, c in enumerate(choices):
                for _ in range(weights[i]):
                    weighted_choices.append(c)

            none_weight = 50
            townhall_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
            if not self.can_afford(townhall_type):
                none_weight += 50
            if self.townhalls.amount < self.MIN_EXPENSION:
                none_weight = 100

            weighted_choices.extend([None]*none_weight)
            class_choice = random.choice(weighted_choices)
            # print(class_choice)

            none_choice = len(list(self.MILITARY_UNIT_CLASS.keys()))
            if class_choice is not None:
                check = await self.create_unit(class_choice, n=1)
                choice = list(self.MILITARY_UNIT_CLASS.keys()).index(class_choice)  # if check else none_choice
            else:
                choice = none_choice

        y = np.zeros(len(self.MILITARY_UNIT_CLASS)+1)
        y[choice] = 1
        # print(f"unit choice: {y}") if self.debug else 0
        self.training_data["create_units_data"].append([self.get_units_state(), y])

    def get_units_state(self):
        state = [self.minerals, self.vespene, self.supply_left, self.expend_count,
                 self.killed_score, self.last_killed_score, self.is_attacking_ratio]
        for unit_class, info in self.MILITARY_UNIT_CLASS.items():
            class_state = [info["supply"], info["created"], info["dead"], info["mineral_cost"], info["vespene_cost"]]
            state.extend(class_state)
        # print(np.array(state))
        return np.array(state)

    def get_enemy_dead_units(self):
        dead_enemies = self.state.dead_units
        for unit in self.units.tags_in(self.state.dead_units):
            dead_enemies.remove(unit.tag)
        self.total_enemy_killed += len(dead_enemies)
        self.last_iteration_enemy_killed = len(dead_enemies)
        # print(self.total_enemy_killed, self.state.score.killed_value_units)

    def update_killed_score(self):
        self.last_killed_score = self.killed_score
        self.killed_score = self.state.score.killed_value_units

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

        if self.scout_group.amount > 0 and self.iteration > self.do_something_after_scout:
            # self.create_scout_points()
            # for i, scout in enumerate(self.scout_group):
            #     if scout.is_idle and not scout.is_moving:
            #         move_to = self.random_location_variance(self.scout_points[i], variance=75)
            #         await self.do(scout.move(move_to))
            for i, scout in enumerate(self.scout_group.idle):
                if scout.is_idle and not scout.is_moving:

                    move_to = self.random_location_variance(self.get_scouting_point(), variance=75)
                    await self.do(scout.move(move_to))
                    # print("oh shite I'm moving")

            self.do_something_after_scout = self.iteration + self.iteration_per_scouting_point

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

    def get_scouting_point(self):
        scouting_points = list(set(self.expansion_locations.keys()) - set(self.owned_expansions.keys()))

        if not scouting_points:
            scouting_points = self.enemy_start_locations + [self.start_location]
        pos = random.choice(scouting_points)
        return pos

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

            army_size_ratio = self.attack_group.amount / self.MAX_SUPPLY
            if army_size_ratio > 1.0:
                army_size_ratio = 1.0

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

            cv2.line(game_data, (0, 23), (int(line_max * army_size_ratio), 23), (200, 200, 200), 3)  # army_size/supplymax ratio
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

    async def distribute_workers(self):
        """
        Distributes workers across all the bases taken.
        WARNING: This is quite slow when there are lots of workers or multiple bases.
        """

        # TODO:
        # OPTIMIZE: Assign idle workers smarter
        # OPTIMIZE: Never use same worker mutltiple times
        owned_expansions = self.owned_expansions
        worker_pool = []
        actions = []

        for idle_worker in self.workers.idle:
            mf = self.state.mineral_field.closest_to(idle_worker)
            actions.append(idle_worker.gather(mf))

        for location, townhall in owned_expansions.items():
            workers = self.workers.closer_than(20, location)
            actual = townhall.assigned_harvesters
            ideal = townhall.ideal_harvesters
            excess = actual - ideal
            if actual > ideal:
                worker_pool.extend(workers.random_group_of(min(excess, len(workers))))
                continue
        for g in self.geysers:
            workers = self.workers.closer_than(5, g)
            actual = g.assigned_harvesters
            ideal = g.ideal_harvesters
            excess = actual - ideal
            if actual > ideal:
                worker_pool.extend(workers.random_group_of(min(excess, len(workers))))
                continue

        for g in self.geysers:
            actual = g.assigned_harvesters
            ideal = g.ideal_harvesters
            deficit = ideal - actual

            for _ in range(deficit):
                if worker_pool:
                    w = worker_pool.pop()
                    if len(w.orders) == 1 and w.orders[0].ability.id is AbilityId.HARVEST_RETURN:
                        actions.append(w.move(g))
                        actions.append(w.return_resource(queue=True))
                    else:
                        actions.append(w.gather(g))

        for location, townhall in owned_expansions.items():
            actual = townhall.assigned_harvesters
            ideal = townhall.ideal_harvesters

            deficit = ideal - actual
            for x in range(0, deficit):
                if worker_pool:
                    w = worker_pool.pop()
                    mf = self.state.mineral_field.closest_to(townhall)
                    if len(w.orders) == 1 and w.orders[0].ability.id is AbilityId.HARVEST_RETURN:
                        actions.append(w.move(townhall))
                        actions.append(w.return_resource(queue=True))
                        actions.append(w.gather(mf, queue=True))
                    else:
                        actions.append(w.gather(mf))

        await self.do_actions(actions)

    def compute_units_efficacity(self):
        threshold = 0.1
        if not self.iteration % self.hm_iteration_per_efficacity_compute:
            incompetences = list()
            for unit_class, info in self.MILITARY_UNIT_CLASS.items():
                total_cost = info["supply"] + info["mineral_cost"] + info["vespene_cost"]
                incompetence = info["dead_batch"] / (max(1, info["created_batch"]))
                incompetences.append(incompetence)

            for i, (unit_class, info) in enumerate(self.MILITARY_UNIT_CLASS.items()):
                be_attacked = False
                if info["attack_ratio"]/self.hm_iteration_per_efficacity_compute >= threshold:
                    be_attacked = True
                    incompetence = incompetences[i] / max(1, incompetences[i], max(incompetences))
                    info["efficacity"] = np.mean([info["efficacity"], (1 - incompetence)])

                info["dead_batch"] = 0
                info["created_batch"] = 0
                if self.debug:
                    print(f"{unit_class} efficacity: {info['efficacity']}, be_attacked: {be_attacked}")
        else:
            for i, (unit_class, info) in enumerate(self.MILITARY_UNIT_CLASS.items()):
                info["attack_ratio"] += self.get_is_attacking_ratio(self.units(unit_class).ready)

    async def put_ability_on_damage_unit(self):
        for unit_type, ability in self.DAMAGE_UNITS_ABILITIES.items():
            try:
                for unit in self.units(unit_type).ready:
                    # if unit.is_attacking:
                    abilities = await self.get_available_abilities(unit)
                    if unit.shield_percentage < 1 and ability in abilities:
                        await self.do(unit(ability))
            except Exception:
                pass

    async def put_ability_on_attacking_unit(self):
        for unit_type, ability in self.ATTACK_UNITS_ABILITIES.items():
            try:
                for unit in self.units(unit_type).ready:
                    abilities = await self.get_available_abilities(unit)
                    try:
                        if unit.is_attacking and ability in abilities:
                            await self.do(unit(ability))
                    except Exception:
                        continue
            except Exception:
                pass


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
