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

    MILITARY_UNIT_CLASS = {
                           UnitTypeId.STALKER: {"max": 20, "priority": 100, "maker_class": UnitTypeId.GATEWAY,
                                                "supply": 2, "created": 0, "dead": 0, "mineral_cost": 125,
                                                "vespene_cost": 50, "efficacity": 1, "created_batch": 0,
                                                "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                "ability_id": AbilityId.GATEWAYTRAIN_STALKER},

                           UnitTypeId.ADEPT: {"max": 6, "priority": 90, "maker_class": UnitTypeId.GATEWAY, "supply": 2,
                                              "created": 0, "dead": 0, "mineral_cost": 100, "vespene_cost": 25,
                                              "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat",
                                              "attack_ratio": 0.0, "ability_id": AbilityId.TRAIN_ADEPT},

                           UnitTypeId.VOIDRAY: {"max": 35, "priority": 150, "maker_class": UnitTypeId.STARGATE,
                                                "supply": 4, "created": 0, "dead": 0, "mineral_cost": 250,
                                                "vespene_cost": 150, "efficacity": 1,  "created_batch": 0,
                                                "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                "ability_id": AbilityId.STARGATETRAIN_VOIDRAY},

                           UnitTypeId.COLOSSUS: {"max": 5, "priority": 150, "maker_class": UnitTypeId.ROBOTICSFACILITY,
                                                 "supply": 6, "created": 0, "dead": 0,
                                                 "mineral_cost": 300, "vespene_cost": 200, "efficacity": 1,
                                                 "created_batch": 0, "dead_batch": 0, "type": "combat",
                                                 "attack_ratio": 0.0,
                                                 "ability_id": AbilityId.ROBOTICSFACILITYTRAIN_COLOSSUS},

                           UnitTypeId.IMMORTAL: {"max": 10, "priority": 110, "maker_class": UnitTypeId.ROBOTICSFACILITY,
                                                 "supply": 6, "created": 0, "dead": 0, "mineral_cost": 275,
                                                 "vespene_cost": 100, "efficacity": 1, "created_batch": 0,
                                                 "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                 "ability_id": AbilityId.ROBOTICSFACILITYTRAIN_IMMORTAL},

                           UnitTypeId.CARRIER: {"max": 3, "priority": 80, "maker_class": UnitTypeId.STARGATE,
                                                "supply": 6, "created": 0, "dead": 0, "mineral_cost": 350,
                                                "vespene_cost": 250, "efficacity": 1, "created_batch": 0,
                                                "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                "ability_id": AbilityId.STARGATETRAIN_CARRIER},

                           UnitTypeId.MOTHERSHIP: {"max": 1, "priority": 200, "maker_class": UnitTypeId.NEXUS,
                                                   "supply": 8, "created": 0, "dead": 0, "mineral_cost": 400,
                                                   "vespene_cost": 400, "efficacity": 1, "created_batch": 0,
                                                   "dead_batch": 0, "type": "support", "attack_ratio": 0.0,
                                                   "ability_id": AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP},

                           UnitTypeId.SENTRY: {"max": 6, "priority": 100, "maker_class": UnitTypeId.GATEWAY, "supply": 2,
                                               "created": 0, "dead": 0, "mineral_cost": 50, "vespene_cost": 100,
                                               "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "support",
                                               "attack_ratio": 0.0, "ability_id": AbilityId.GATEWAYTRAIN_SENTRY},

                           UnitTypeId.TEMPEST: {"max": 5, "priority": 100, "maker_class": UnitTypeId.STARGATE,
                                                "supply": 5, "created": 0, "dead": 0, "mineral_cost": 250,
                                                "vespene_cost": 175, "efficacity": 1, "created_batch": 0,
                                                "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                "ability_id": AbilityId.STARGATETRAIN_TEMPEST},

                           UnitTypeId.ZEALOT: {"max": 6, "priority": 90, "maker_class": UnitTypeId.GATEWAY,
                                               "supply": 2, "created": 0, "dead": 0, "mineral_cost": 100,
                                               "vespene_cost": 0, "efficacity": 1, "created_batch": 0,
                                               "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                               "ability_id": AbilityId.GATEWAYTRAIN_ZEALOT},

                           UnitTypeId.DISRUPTOR: {"max": 5, "priority": 100, "maker_class": UnitTypeId.ROBOTICSFACILITY,
                                                  "supply": 3, "created": 0, "dead": 0, "mineral_cost": 150,
                                                  "vespene_cost": 150, "efficacity": 1, "created_batch": 0,
                                                  "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                  "ability_id": AbilityId.TRAIN_DISRUPTOR},

                           UnitTypeId.DARKTEMPLAR: {"max": 4, "priority": 100, "maker_class": UnitTypeId.GATEWAY,
                                                    "supply": 2, "created": 0, "dead": 0, "mineral_cost": 125,
                                                    "vespene_cost": 125, "efficacity": 1, "created_batch": 0,
                                                    "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                    "ability_id": AbilityId.GATEWAYTRAIN_DARKTEMPLAR},

                           UnitTypeId.HIGHTEMPLAR: {"max": 4, "priority": 100, "maker_class": UnitTypeId.GATEWAY,
                                                    "supply": 2, "created": 0, "dead": 0, "mineral_cost": 50,
                                                    "vespene_cost": 150, "efficacity": 1, "created_batch": 0,
                                                    "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                                                    "ability_id": AbilityId.GATEWAYTRAIN_HIGHTEMPLAR},

                           # UnitTypeId.ARCHON: {"max": 4, "priority": 100, "maker_class": UnitTypeId.DARKTEMPLAR,
                           #                     "supply": 2, "created": 0, "dead": 0, "mineral_cost": 50,
                           #                     "vespene_cost": 150, "efficacity": 1, "created_batch": 0,
                           #                     "dead_batch": 0, "type": "combat", "attack_ratio": 0.0,
                           #                     "ability_id": AbilityId.MORPH_ARCHON},
                           }

    Q_CLASS = {UnitTypeId.STALKER: {"max": 0, "priority": 100, "maker_class": UnitTypeId.WARPGATE, "supply": 2}}

    SCOUT_CLASS = {UnitTypeId.OBSERVER: {"max": 10, "priority": 1, "maker_class": UnitTypeId.ROBOTICSFACILITY,
                                         "supply": 1}}

    MEDIC_CLASS = {}

    CIVIL_UNIT_CLASS = {UnitTypeId.PROBE: {"max": 70, "priority": 1,
                                           "maker_class": UnitTypeId.NEXUS, "supply": 1}}

    CIVIL_BUILDING_CLASS = {UnitTypeId.NEXUS: {"type": "main"},
                            UnitTypeId.ASSIMILATOR: {"type": "vaspene"},
                            UnitTypeId.PYLON: {"type": "supply"}}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.CYBERNETICSCORE: {"priority": 1, "max": 1,
                                                             "avg_per_cmdc": 1, "add_on": [],
                                                             "upgrade": [UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
                                                                         UpgradeId.PROTOSSAIRARMORSLEVEL1,
                                                                         UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
                                                                         UpgradeId.PROTOSSAIRARMORSLEVEL2,
                                                                         UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
                                                                         UpgradeId.PROTOSSAIRARMORSLEVEL3],
                                                             "ability_id": AbilityId.PROTOSSBUILD_CYBERNETICSCORE},
                                UnitTypeId.GATEWAY: {"priority": 1, "max": 4,
                                                     "avg_per_cmdc": 2, "add_on": [],
                                                     "upgrade": [],
                                                     "ability_id": AbilityId.PROTOSSBUILD_GATEWAY},

                                UnitTypeId.ROBOTICSFACILITY: {"priority": 1, "max": 1,
                                                              "avg_per_cmdc": 1, "add_on": [],
                                                              "upgrade": [],
                                                              "ability_id": AbilityId.PROTOSSBUILD_ROBOTICSFACILITY},

                                UnitTypeId.STARGATE: {"priority": 1, "max": 4,
                                                      "avg_per_cmdc": 2, "add_on": [],
                                                      "upgrade": [],
                                                      "ability_id": AbilityId.PROTOSSBUILD_STARGATE},

                                UnitTypeId.FORGE: {"priority": 1, "max": 1,
                                                   "avg_per_cmdc": 1, "add_on": [],
                                                   "upgrade": [UpgradeId.PROTOSSSHIELDSLEVEL1,
                                                               UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
                                                               UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                                                               UpgradeId.PROTOSSSHIELDSLEVEL2,
                                                               UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
                                                               UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                                               UpgradeId.PROTOSSSHIELDSLEVEL3,
                                                               UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
                                                               UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3],
                                                   "ability_id": AbilityId.PROTOSSBUILD_FORGE},

                                UnitTypeId.ROBOTICSBAY: {"priority": 1, "max": 1,
                                                         "avg_per_cmdc": 1, "add_on": [],
                                                         "upgrade": [UpgradeId.EXTENDEDTHERMALLANCE,
                                                                     UpgradeId.OBSERVERGRAVITICBOOSTER],
                                                         "ability_id": AbilityId.PROTOSSBUILD_ROBOTICSBAY},

                                UnitTypeId.FLEETBEACON: {"priority": 1, "max": 1,
                                                         "avg_per_cmdc": 1, "add_on": [], "upgrade": [],
                                                         "ability_id": AbilityId.PROTOSSBUILD_FLEETBEACON},

                                UnitTypeId.TWILIGHTCOUNCIL: {"priority": 1, "max": 1,
                                                             "avg_per_cmdc": 1, "add_on": [],
                                                             "upgrade": [],
                                                             "ability_id": AbilityId.PROTOSSBUILD_TWILIGHTCOUNCIL},

                                UnitTypeId.DARKSHRINE: {"priority": 1, "max": 1,
                                                        "avg_per_cmdc": 1, "add_on": [],
                                                        "upgrade": [],
                                                        "ability_id": AbilityId.PROTOSSBUILD_DARKSHRINE},

                                UnitTypeId.TEMPLARARCHIVE: {"priority": 1, "max": 1,
                                                            "avg_per_cmdc": 1, "add_on": [],
                                                            "upgrade": [],
                                                            "ability_id": AbilityId.PROTOSSBUILD_TEMPLARARCHIVE}
                                }

    DEFENSE_BUILDING_CLASS = {UnitTypeId.PHOTONCANNON: {"priority": 1, "max": 20,
                                                        "avg_per_cmdc": 3, "add_on": [],
                                                        "upgrade": [],
                                                        "ability_id": AbilityId.PROTOSSBUILD_PHOTONCANNON}}

    RESERVE_BUILDING_CLASS = {UnitTypeId.GATEWAY: {"priority": 1, "max": 1,
                                                   "avg_per_cmdc": 1, "add_on": [],
                                                   "upgrade": [], "ability_id": AbilityId.PROTOSSBUILD_GATEWAY}}

    DAMAGE_UNITS_ABILITIES = {UnitTypeId.SENTRY: AbilityId.GUARDIANSHIELD_GUARDIANSHIELD,
                              UnitTypeId.IMMORTAL: AbilityId.EFFECT_IMMORTALBARRIER,
                              UnitTypeId.VOIDRAY: AbilityId.EFFECT_VOIDRAYPRISMATICALIGNMENT}

    RATIO_DEF_ATT_UNITS = 0.0
    MIN_ARMY_SIZE_FOR_ATTACK = 30
    MIN_ARMY_SIZE_FOR_RETRAITE = 5

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True, epsilon=0.05):
        super(JarexProtoss, self).__init__(use_model, human_control, debug, take_training_data, epsilon)

        self.MAX_SUPPLY_LEFT = 20

        self.warp_gate_created = False

    async def on_step(self, iteration):
        await super(JarexProtoss, self).on_step(iteration)

        if self.warp_gate_created:
            await self.build_warp_gate()
        else:
            await self.init_warp_gate()

        try:
            await self.research_other_update()
        except Exception:
            pass

    async def create_military_buildings(self):
        supply_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")

        supps = self.units(supply_type).ready
        if self.townhalls.amount <= 0 or supps.amount <= 0:
            return None

        for b_class, info in self.MILITARY_BUILDINGS_CLASS.items():
            pos = supps.random.position
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
                                         max_distance=30, placement_step=(10 if info["add_on"] else 5))
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

    async def build_defense_buildings(self):
        try:
            if not self.townhalls.amount and not self.workers.amount:
                return None

            for b_class, info in self.DEFENSE_BUILDING_CLASS.items():
                pos = self.units(UnitTypeId.PYLON).closest_to(random.choice(self.enemy_start_locations)).position
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

    async def create_supply_depot(self):
        cmdc_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "main")
        supply_depot_type = self.find_key_per_info(self.CIVIL_BUILDING_CLASS, "type", "supply")
        cmdcs = self.units(cmdc_type)
        try:
            if self.supply_cap < self.MAX_SUPPLY \
                    and (not self.units(supply_depot_type).amount or self.expend_count > 1):
                if self.supply_left < self.MAX_SUPPLY_LEFT and self.can_afford(supply_depot_type) \
                        and cmdcs.amount and not self.already_pending(supply_depot_type):
                    pos = self.random_location_variance(cmdcs.random.position.towards(self.game_info.map_center, 5), 10)
                    await self.build(supply_depot_type, near=pos, max_distance=10)
        except ValueError:
            pass

    async def create_unit(self, unit_class, info=None, n=1) -> bool:
        check = False

        if info is None:
            info = self.MILITARY_UNIT_CLASS[unit_class]

        units = self.units(unit_class)
        makers = self.units(info["maker_class"]).ready
        pylons = self.units(UnitTypeId.PYLON).ready
        if units.amount < info["max"] and self.supply_left >= info["supply"] and makers.amount > 0:
            for _ in range(n):
                makers = makers.noqueue
                if self.can_afford(unit_class) and makers.amount > 0:
                    if info["maker_class"] == UnitTypeId.WARPGATE:
                        if pylons:
                            proxy = pylons.closest_to(random.choice(self.enemy_start_locations))
                            check = await self.warp_new_units(proxy, unit_class)
                    else:
                        if makers.amount > 0:
                            maker = makers.random
                            abilities = await self.get_available_abilities(maker)
                            try:
                                if info["ability_id"] in abilities:
                                    self.current_actions.append(maker.train(unit_class))
                                    check = True
                            except Exception:
                                continue
        return check

    async def init_warp_gate(self):
        cyberneticscore = self.units(UnitTypeId.CYBERNETICSCORE).ready.noqueue
        if cyberneticscore.amount:
            cyberneticscore = cyberneticscore.random
        if cyberneticscore and self.can_afford(UpgradeId.WARPGATERESEARCH):
            try:
                abilities = await self.get_available_abilities(cyberneticscore)
                if AbilityId.RESEARCH_WARPGATE in abilities:
                    await self.do(cyberneticscore(AbilityId.RESEARCH_WARPGATE))

                    info = self.MILITARY_BUILDINGS_CLASS[UnitTypeId.GATEWAY]
                    self.MILITARY_BUILDINGS_CLASS[UnitTypeId.WARPGATE] = info
                    del self.MILITARY_BUILDINGS_CLASS[UnitTypeId.GATEWAY]

                    for unit_type, info in self.MILITARY_UNIT_CLASS.items():
                        if info["maker_class"] == UnitTypeId.GATEWAY:
                            info["maker_class"] = UnitTypeId.WARPGATE
                        self.MILITARY_UNIT_CLASS[unit_type] = info

                    self.warp_gate_created = True
            except Exception as e:
                pass

    async def build_warp_gate(self):
        try:
            if self.units(UnitTypeId.PYLON).ready.exists:
                pylon = self.units(UnitTypeId.PYLON).ready.random
                if self.can_afford(UnitTypeId.GATEWAY) and \
                        self.units(UnitTypeId.WARPGATE).amount < self.MILITARY_BUILDINGS_CLASS[UnitTypeId.WARPGATE][
                    "max"] \
                        and not self.already_pending(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylon)
        except Exception:
            pass

    async def warp_new_units(self, proxy, unit_class) -> bool:
        check = False

        for warpgate in self.units(UnitTypeId.WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            # all the units have the same cooldown anyway so let's just look at STALKER
            if AbilityId.WARPGATETRAIN_STALKER in abilities:
                pos = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    pylons = self.units(UnitTypeId.PYLON).ready
                    if pylons:
                        pos = pylons.random.position.to2.random_on_distance(4)
                        placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    return check
                await self.do(warpgate.warp_in(unit_class, placement))
                check = True

        return check

    async def research_other_update(self):
        twilight_councils = self.units(UnitTypeId.TWILIGHTCOUNCIL).ready
        if twilight_councils:
            twilight_council = twilight_councils.random
            try:
                abilities = await self.get_available_abilities(twilight_council)
                if AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES in abilities:
                    await self.do(twilight_council(AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES))
                elif AbilityId.RESEARCH_CHARGE in abilities:
                    await self.do(twilight_council(AbilityId.RESEARCH_CHARGE))
            except Exception:
                pass

        robotics_bays = self.units(UnitTypeId.ROBOTICSBAY).ready
        if robotics_bays:
            robotics_bay = robotics_bays.random
            try:
                abilities = await self.get_available_abilities(robotics_bay)
                if AbilityId.RESEARCH_EXTENDEDTHERMALLANCE in abilities:
                    await self.do(robotics_bay(AbilityId.RESEARCH_EXTENDEDTHERMALLANCE))
            except Exception:
                pass


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot
    from JarexTerran import JarexTerran

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=False, human_control=False,
                                               debug=True, take_training_data=False, epsilon=1.0),
            name=JarexProtoss.BOTNAME),
        Computer(Race.Protoss, Difficulty.Hard)
    ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=True, human_control=False,
    #                                            debug=False, take_training_data=False),
    #         name=JarexProtoss.BOTNAME),
    #     Bot(JarexTerran.BOTRACE, JarexTerran(use_model=False, human_control=False,
    #                                          debug=False, take_training_data=False),
    #         name=JarexTerran.BOTNAME)
    # ], realtime=True)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=True, human_control=False,
    #                                            debug=False, take_training_data=False),
    #         name=JarexProtoss.BOTNAME+"_deep_model"),
    #     Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=False, human_control=False,
    #                                            debug=False, take_training_data=False),
    #         name=JarexProtoss.BOTNAME+"_stats_model")
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Human(Race.Protoss, name="Cortex"),
    #     Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=False, human_control=False,
    #                                            debug=False, take_training_data=False, epsilon=1.0),
    #         name=JarexProtoss.BOTNAME)
    # ], realtime=True)
