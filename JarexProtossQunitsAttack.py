from JarexProtoss import JarexProtoss


class JarexProtossQUnitsAttack(JarexProtoss):
    BOTNAME = "JarexProtossQUnitsAttack"

    def __init__(self):
        super().__init__(use_model=False, human_control=False, debug=True, take_training_data=False, epsilon=1.0)

    async def on_unit_created(self, unit):
        if unit.type_id in self.SCOUT_CLASS and not self.scout_group_is_complete():
            self.scout_group.append(unit)
        elif unit.type_id in self.Q_CLASS and not self.q_group_is_complete():
            self.q_group.append(unit)
        elif unit.type_id in self.MILITARY_UNIT_CLASS:
            self.q_group.append(unit)

        # print(f"q_group size: {self.q_group.amount}")

        if unit.type_id in self.MILITARY_UNIT_CLASS:
            self.MILITARY_UNIT_CLASS[unit.type_id]["created"] += 1
            self.MILITARY_UNIT_CLASS[unit.type_id]["created_batch"] += 1


if __name__ == '__main__':
    import sc2
    from sc2.player import Bot, Computer, Human
    from sc2 import Race, Difficulty
    import random

    hm_game = 1
    win_counter = 0
    difficulty = Difficulty.Harder
    races = [Race.Zerg, Race.Terran, Race.Protoss]
    ennemie_is_stats_model = True

    for _ in range(hm_game):
        result = sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
            Bot(JarexProtossQUnitsAttack.BOTRACE, JarexProtossQUnitsAttack(),
                name=JarexProtossQUnitsAttack.BOTNAME),
            Bot(JarexProtoss.BOTRACE, JarexProtoss(use_model=False, human_control=False,
                                                   debug=False, take_training_data=True, epsilon=1.0),
                name=JarexProtoss.BOTNAME)
            if ennemie_is_stats_model
            else Computer(random.choice(races), difficulty)
        ], realtime=False)

        if result == sc2.Result.Victory:
            win_counter += 1

    print(f"--- Training on {difficulty} Finished ---")
    print(f"win_counter: {win_counter}, game_counter: {hm_game}")
    print(f"ratio of game win: {win_counter / hm_game}")
    print(f"-" * 175)
