import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, ZEALOT
import random


class TutoBot(sc2.BotAI):
    async def on_step(self, iteration):
        # we build some workers
        for nexus in self.units(NEXUS).ready.noqueue:
            if self.can_afford(PROBE):
                await self.do(nexus.train(PROBE))

        # we build some pylon to have more units
        if not self.supply_left and not self.already_pending(PYLON):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON):
                    await self.build(PYLON, near=nexuses.first)

        # we build some assimilator to take other ressources
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))

        # we build some "barracks"
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if not len(self.units(GATEWAY)) and self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
                await self.build(GATEWAY, near=pylon)

        # we do somme attackers
        for gw in self.units(GATEWAY).ready.noqueue:
            if self.can_afford(ZEALOT) and self.supply_left > 0:
                await self.do(gw.train(ZEALOT))

        # we defend our base
        if self.units(ZEALOT).amount:
            if len(self.known_enemy_units):
                for s in self.units(ZEALOT).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))


def main():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, TutoBot()),
        Computer(Race.Terran, Difficulty.Easy)
        ], realtime=False)


if __name__ == '__main__':
    main()
