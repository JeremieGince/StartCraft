from sc2 import maps
from sc2.main import run_game

from JarexTerran import JarexTerran

from sc2.data import Race, Difficulty
from sc2.constants import *
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.player import Bot, Computer
from JarexSc2 import JarexSc2


class MarineOverflow(JarexTerran):
    BOTNAME = "MarineOverflow"
    MIN_ARMY_SIZE_FOR_ATTACK = 25
    RATIO_DEF_ATT_UNITS = 0.05

    MILITARY_UNIT_CLASS = {UnitTypeId.MARINE: {"max": 118, "priority": 50, "maker_class": UnitTypeId.BARRACKS,
                                               "supply": 1, "created": 0, "dead": 0, "mineral_cost": 125,
                                               "vespene_cost": 50, "efficacity": 1, "created_batch": 0,
                                               "dead_batch": 0, "type": "combat", "attack_ratio": 0.0},
                           UnitTypeId.MEDIVAC: {"max": 6, "priority": 1, "maker_class": UnitTypeId.STARPORT,
                                                "supply": 6, "created": 0, "dead": 0, "mineral_cost": 125,
                                                "vespene_cost": 50, "efficacity": 1, "created_batch": 0,
                                                "dead_batch": 0, "type": "support", "attack_ratio": 0.0}}

    MILITARY_BUILDINGS_CLASS = {UnitTypeId.BARRACKS: {"priority": 1, "max": 10,
                                                      "avg_per_cmdc": 3, "add_on": [],
                                                      "upgrade": []},
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
                                UnitTypeId.STARPORT: {"priority": 1, "max": 1,
                                                      "avg_per_cmdc": 1, "add_on": [],
                                                      "upgrade": []}
                                }

    def __init__(self, use_model=False, human_control=False, debug=False, take_training_data=True, epsilon=0.05):
        super(MarineOverflow, self).__init__(use_model, human_control, debug, take_training_data, epsilon)
        # faire le scout group, faire un trucs que c'est moi qui fait les choix pendant la game ou random ou par models.
        # faire l'affichage de données, finir d'écouter les tuto pour avoir plus d'idées

    async def on_end(self, game_result):
        # print("Ennemy killed: ", self._game_info.killed_enemy)
        return JarexTerran.on_end(self, game_result)

    async def create_military_buildings(self):
        # await super(JarexTerran, self).create_military_buildings()

        await JarexSc2.create_military_buildings(self)

        barracks = self.units(UnitTypeId.BARRACKS).ready
        if barracks:
            barrack = barracks.random
            if barrack.add_on_tag == 0 and self.can_afford(UnitTypeId.BARRACKSTECHLAB) \
                    and not self.already_pending(UnitTypeId.BARRACKSTECHLAB) \
                    and not self.units(UnitTypeId.BARRACKSTECHLAB):
                try:
                    abilities = await self.get_available_abilities([barrack])
                    if AbilityId.BUILD_TECHLAB_BARRACKS in abilities:
                        self.do(barrack(AbilityId.BUILD_TECHLAB_BARRACKS))
                except Exception:
                    return None

        starports = self.units(UnitTypeId.STARPORT).ready
        if starports:
            starport = starports.random
            if starport.add_on_tag == 0 and self.can_afford(UnitTypeId.STARPORTTECHLAB) \
                    and not self.already_pending(UnitTypeId.STARPORTTECHLAB) \
                    and not self.units(UnitTypeId.STARPORTTECHLAB):
                try:
                    abilities = await self.get_available_abilities([starport])
                    if AbilityId.BUILD_TECHLAB_STARPORT in abilities:
                        self.do(starport(AbilityId.BUILD_TECHLAB_STARPORT))
                except Exception:
                    return None


if __name__ == '__main__':
    import os

    os.environ["SC2PATH"] = open("SC2PATH.txt").read().rstrip("\n")
    # from examples.terran.proxy_rax import ProxyRaxBot
    # from Sentdex_tuto.t6_defeated_hard_AI import SentdeBot

    run_game(maps.get("AbyssalReefLE"), [
        Bot(
            MarineOverflow.BOTRACE,
            MarineOverflow(
                use_model=False,
                human_control=False,
                debug=True,
                take_training_data=False
            )
        ),
        Computer(Race.Zerg, Difficulty.Hard)
    ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(Race.Terran, JarexTerran()),
    #     Bot(Race.Protoss, SentdeBot())
    # ], realtime=False)

    # sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
    #     Bot(MarineOverflow.BOTRACE, MarineOverflow()),
    #     Bot(JarexTerran.BOTRACE, JarexTerran(False, False))
    # ], realtime=False)
