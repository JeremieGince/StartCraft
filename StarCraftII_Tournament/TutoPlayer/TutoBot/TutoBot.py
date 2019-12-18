import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.units import UnitTypeId
import random


class TutoBot(sc2.BotAI):
    BOTNAME = "TutoBot"
    BOTRACE = Race.Protoss

    async def on_step(self, iteration):
        # we build 16 workers
        if self.units(UnitTypeId.PROBE).amount < 16:
            for nexus in self.units(UnitTypeId.NEXUS).ready.noqueue:
                if self.can_afford(UnitTypeId.PROBE):
                    await self.do(nexus.train(UnitTypeId.PROBE))

        # we build some pylon to have more units
        if not self.supply_left and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    await self.build(UnitTypeId.PYLON, near=nexuses.first)

        await self.create_vaspene()

        # we build some "barracks"
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random
            if not len(self.units(UnitTypeId.GATEWAY)) and self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
                await self.build(UnitTypeId.GATEWAY, near=pylon)

        # we do somme attackers
        for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue:
            if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0:
                await self.do(gw.train(UnitTypeId.ZEALOT))

        # we defend our base
        if self.units(UnitTypeId.ZEALOT).amount:
            if len(self.known_enemy_units):
                for s in self.units(UnitTypeId.ZEALOT).idle:
                    await self.do(s.take_action(random.choice(self.known_enemy_units)))

        # every 1000 game step we expand (build a new nexus)
        if not iteration % 1000:
            await self.expand_now()

    async def create_vaspene(self):
        # we build some assimilator to take other ressources
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))


if __name__ == '__main__':

    # Notez que lors du premier run, deux erreurs l'un après l'autre interrompera le code.
    # Le premier vient de assert self.id != 0 à la ligne 93 de game_data, commentez le.
    # Le deuxième vient de assert self.bits_per_pixel % 8 == 0, "Unsupported pixel density" à la ligne 9 de pixel_map,
    # commentez le.

    run_game(maps.get("AbyssalReefLE"), [
        Bot(TutoBot.BOTRACE, TutoBot()),
        Computer(Race.Zerg, Difficulty.Easy)
    ], realtime=False)
