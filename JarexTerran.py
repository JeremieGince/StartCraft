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

    MILITARY_UNIT_CLASS = {MARINE: {"max": 50, "priority": 100, "maker_class": BARRACKS, "supply": 1,
                                    "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50, "efficacity": 1,
                                    "created_batch": 0, "dead_batch": 0, "type": "combat", "attack_ratio": 0.0},

                           SIEGETANK: {"max": 25, "priority": 100, "maker_class": FACTORY, "supply": 4, "created": 0,
                                       "dead": 0, "mineral_cost": 125, "vespene_cost": 50, "efficacity": 1,
                                       "created_batch": 0, "dead_batch": 0, "type": "combat", "attack_ratio": 0.0},

                           UnitTypeId.THOR: {"max": 50, "priority": 100, "maker_class": FACTORY, "supply": 6,
                                             "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                             "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat",
                                             "attack_ratio": 0.0},

                           UnitTypeId.CYCLONE: {"max": 50, "priority": 100, "maker_class": FACTORY, "supply": 2,
                                                "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                                "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat",
                                                "attack_ratio": 0.0},

                           VIKING: {"max": 5, "priority": 90, "maker_class": STARPORT, "supply": 2, "created": 0,
                                    "dead": 0, "mineral_cost": 125, "vespene_cost": 50, "efficacity": 1,
                                    "created_batch": 0, "dead_batch": 0, "type": "combat", "attack_ratio": 0.0},

                           UnitTypeId.BATTLECRUISER: {"max": 15, "priority": 150, "maker_class": STARPORT, "supply": 6,
                                                      "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                                      "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat", "attack_ratio": 0.0},

                           UnitTypeId.MARAUDER: {"max": 15, "priority": 100, "maker_class": BARRACKS, "supply": 2,
                                                 "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                                 "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat",
                                                 "attack_ratio": 0.0},

                           UnitTypeId.GHOST: {"max": 25, "priority": 100, "maker_class": BARRACKS, "supply": 2,
                                              "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                              "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "combat",
                                              "attack_ratio": 0.0},

                           UnitTypeId.MEDIVAC: {"max": 6, "priority": 100, "maker_class": STARPORT, "supply": 6,
                                                "created": 0, "dead": 0, "mineral_cost": 125, "vespene_cost": 50,
                                                "efficacity": 1, "created_batch": 0, "dead_batch": 0, "type": "support",
                                                "attack_ratio": 0.0}}

    SCOUT_CLASS = {UnitTypeId.RAVEN: {"max": 3, "priority": 1, "maker_class": STARPORT, "supply": 1},
                   MARINE: {"max": 4, "priority": 1, "maker_class": BARRACKS, "supply": 1}}

    MEDIC_CLASS = {UnitTypeId.MEDIVAC: {"max": 2, "priority": 50, "maker_class": STARPORT, "supply": 6}}

    CIVIL_UNIT_CLASS = {UnitTypeId.SCV: {"max": 70, "priority": 1,
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
                                                        "upgrade": []},

                                UnitTypeId.ENGINEERINGBAY: {"priority": 1, "max": 1,
                                                            "avg_per_cmdc": 1, "add_on": [],
                                                            "upgrade": [UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                                                                        UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                                                                        UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                                                                        UpgradeId.TERRANINFANTRYARMORSLEVEL3]},

                                }

    DEFENSE_BUILDING_CLASS = {UnitTypeId.MISSILETURRET: {"priority": 1, "max": 15,
                                                         "avg_per_cmdc": 2, "add_on": [], "upgrade": []}}

    RESERVE_BUILDING_CLASS = {UnitTypeId.BARRACKS: {"priority": 1, "max": 1, "avg_per_cmdc": 1, "add_on": [],
                                                    "upgrade": []}}

    # DAMAGE_UNITS_ABILITIES = {UnitTypeId.SIEGETANK: AbilityId.SIEGEBREAKERSIEGE_SIEGEMODE}

    RATIO_DEF_ATT_UNITS = 0.05
    MIN_ARMY_SIZE_FOR_ATTACK = 50

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True, epsilon=0.05):
        super(JarexTerran, self).__init__(use_model, human_control, debug, take_training_data, epsilon)

    async def on_step(self, iteration):
        await super(JarexTerran, self).on_step(iteration)

        try:
            await self.reserch_other_upgrade()
        except Exception:
            pass

        try:
            await self.use_abilities_siege_tank()

        except Exception:
            pass

    async def create_supply_depot(self):
        await super(JarexTerran, self).create_supply_depot()
        for supply_dep in self.units(UnitTypeId.SUPPLYDEPOT).ready:
            abilities = await self.get_available_abilities(supply_dep)
            if AbilityId.MORPH_SUPPLYDEPOT_LOWER in abilities:
                await self.do(supply_dep(AbilityId.MORPH_SUPPLYDEPOT_LOWER))

    async def create_military_buildings(self):
        await super(JarexTerran, self).create_military_buildings()

        barracks = self.units(UnitTypeId.BARRACKS).ready.noqueue
        if barracks:
            barrack = barracks.random
            if barrack.add_on_tag == 0 and self.can_afford(UnitTypeId.BARRACKSTECHLAB) \
                    and not self.already_pending(UnitTypeId.BARRACKSTECHLAB):
                try:
                    abilities = await self.get_available_abilities(barrack)
                    if AbilityId.BUILD_TECHLAB_BARRACKS in abilities:
                        await self.do(barrack(AbilityId.BUILD_TECHLAB_BARRACKS))
                except Exception:
                    return None

        factories = self.units(UnitTypeId.FACTORY).ready.noqueue
        if factories:
            factory = factories.random
            if factory.add_on_tag == 0 and self.can_afford(UnitTypeId.FACTORYTECHLAB) \
                    and not self.already_pending(UnitTypeId.FACTORYTECHLAB):
                try:
                    abilities = await self.get_available_abilities(factory)
                    if AbilityId.BUILD_TECHLAB_FACTORY in abilities:
                        await self.do(factory(AbilityId.BUILD_TECHLAB_FACTORY))
                except Exception:
                    return None

        starports = self.units(UnitTypeId.STARPORT).ready.noqueue
        if starports:
            starport = starports.random
            if starport.add_on_tag == 0 and self.can_afford(UnitTypeId.STARPORTTECHLAB) \
                    and not self.already_pending(UnitTypeId.STARPORTTECHLAB):
                try:
                    abilities = await self.get_available_abilities(starport)
                    if AbilityId.BUILD_TECHLAB_STARPORT in abilities:
                        await self.do(starport(AbilityId.BUILD_TECHLAB_STARPORT))
                except Exception:
                    return None

    async def reserch_other_upgrade(self):
        tech_labs = self.units(UnitTypeId.BARRACKSTECHLAB).ready
        if tech_labs:
            tech_lab = tech_labs.random
            abilities = await self.get_available_abilities(tech_lab)
            if AbilityId.RESEARCH_COMBATSHIELD in abilities:
                await self.do(tech_lab(AbilityId.RESEARCH_COMBATSHIELD))
            elif AbilityId.BARRACKSTECHLABRESEARCH_STIMPACK in abilities:
                await self.do(tech_lab(AbilityId.BARRACKSTECHLABRESEARCH_STIMPACK))

        fusionscore = self.units(UnitTypeId.FUSIONCORE).ready
        if fusionscore:
            fusioncore = fusionscore.random
            abilities = await self.get_available_abilities(fusioncore)
            if AbilityId.YAMATO_YAMATOGUN in abilities:
                await self.do(fusioncore(AbilityId.YAMATO_YAMATOGUN))

    async def use_abilities_siege_tank(self):
        try:
            for unit in self.units(UnitTypeId.SIEGETANK).ready:
                abilities = await self.get_available_abilities(unit)
                try:
                    if unit.is_attacking and AbilityId.SIEGEMODE_SIEGEMODE in abilities and unit.weapon_cooldown:
                        await self.do(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                except Exception:
                    continue
        except Exception:
            pass

        try:
            for unit in self.units(UnitTypeId.SIEGETANKSIEGED).ready:
                abilities = await self.get_available_abilities(unit)
                try:
                    if not unit.is_attacking and AbilityId.UNSIEGE_UNSIEGE in abilities:
                        await self.do(unit(AbilityId.UNSIEGE_UNSIEGE))
                except Exception:
                    continue
        except Exception:
            pass

    async def use_stim_pack(self):
        for marine in self.units(UnitTypeId.MARINE).ready:
            abilities = await self.get_available_abilities(marine)
            try:
                if marine.is_attacking and AbilityId.EFFECT_STIM_MARINE in abilities \
                        and marine.health_percentage >= 0.5 and marine.weapon_cooldown:
                    await self.do(marine(AbilityId.EFFECT_STIM_MARINE))
            except Exception:
                continue


if __name__ == '__main__':
    from examples.terran.proxy_rax import ProxyRaxBot
    from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(Race.Terran, JarexTerran(use_model=False, human_control=False, debug=True, take_training_data=False),
            name=JarexTerran.BOTNAME),
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
