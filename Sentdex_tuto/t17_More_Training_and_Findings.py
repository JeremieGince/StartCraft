import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import cv2
import time


def get_session(gpu_fraction=0.85):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


model = Sequential()
model.add(Conv2D(32, (7, 7), padding='same',
                 input_shape=(176, 200, 1),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))

learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate)#, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE2-{}-{}".format(int(time.time()), learning_rate))

train_data_dir = "train_data"

model = keras.models.load_model('BasicCNN-5000-epochs-0.001-LR-STAGE2')


def check_data(choices):
    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


hm_epochs = 5000

for i in range(hm_epochs):
    current = 0
    increment = 50
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        try:
            print("WORKING ON {}:{}, EPOCH:{}".format(current, current+increment, i))

            choices = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: [],
                       5: [],
                       6: [],
                       7: [],
                       8: [],
                       9: [],
                       10: [],
                       11: [],
                       12: [],
                       13: [],
                       }

            for file in all_files[current:current+increment]:
                try:
                    full_path = os.path.join(train_data_dir, file)
                    data = np.load(full_path)
                    data = list(data)
                    for d in data:
                        choice = np.argmax(d[0])
                        choices[choice].append([d[0], d[1]])
                except Exception as e:
                    print(str(e))

            lengths = check_data(choices)

            lowest_data = min(lengths)

            for choice in choices:
                random.shuffle(choices[choice])
                choices[choice] = choices[choice][:lowest_data]

            check_data(choices)

            train_data = []

            for choice in choices:
                for d in choices[choice]:
                    train_data.append(d)

            random.shuffle(train_data)
            print(len(train_data))

            test_size = 100
            batch_size = 128  # 128 best so far.

            x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 1)
            y_train = np.array([i[0] for i in train_data[:-test_size]])

            x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 1)
            y_test = np.array([i[0] for i in train_data[-test_size:]])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      validation_data=(x_test, y_test),
                      shuffle=True,
                      epochs=1,
                      verbose=1, callbacks=[tensorboard])

            model.save("BasicCNN-5000-epochs-0.001-LR-STAGE2")
        except Exception as e:
            print(str(e))
        current += increment
        if current > maximum:
            not_maximum = False


import tensorflow as tf
import keras.backend.tensorflow_backend as backend
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
from sc2 import position
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, SCV, DRONE, ROBOTICSFACILITY, OBSERVER, \
 ZEALOT, STALKER
import random
import cv2
import numpy as np
import os
import time
import math
import keras

#os.environ["SC2PATH"] = '/home/paperspace/Desktop/testing_model/StarCraftII/'
HEADLESS = False


def get_session(gpu_fraction=0.3):
    """Assume that you have 6GB of GPU memory and want to allocate ~2GB"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
backend.set_session(get_session())


class SentdeBot(sc2.BotAI):
    def __init__(self, use_model=False, title=1):
        self.MAX_WORKERS = 50
        self.do_something_after = 0
        self.use_model = use_model
        self.title = title

        ###############################
        # DICT {UNIT_ID:LOCATION}
        # every iteration, make sure that unit id still exists!
        self.scouts_and_spots = {}

        # ADDED THE CHOICES #
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_stargate,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.attack_known_enemy_unit,
                        11: self.attack_known_enemy_structure,
                        12: self.expand,  # might just be self.expand_now() lol
                        13: self.do_nothing,
                        }

        self.train_data = []
        if self.use_model:
            print("USING MODEL!")
            self.model = keras.models.load_model("STAGE2V2")


    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)

        if self.use_model:
            with open("gameout-model-vs-easy.txt","a") as f:
                f.write("Model {} - {}\n".format(game_result, int(time.time())))

    async def on_step(self, iteration):

        self.time = (self.state.game_loop/22.4) / 60
        #print('Time:',self.time)

        if iteration % 5 == 0:
            await self.distribute_workers()
        await self.scout()
        await self.intel()
        await self.do_something()

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]

        #  FIXED THIS
        x += random.randrange(-5,5)
        y += random.randrange(-5,5)

        if x < 0:
            print("x below")
            x = 0
        if y < 0:
            print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            print("y above")
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))

        return go_to


    async def scout(self):
        '''
        ['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_game_data', '_proto', '_type_data', 'add_on_tag', 'alliance', 'assigned_harvesters', 'attack', 'build', 'build_progress', 'cloak', 'detect_range', 'distance_to', 'energy', 'facing', 'gather', 'has_add_on', 'has_buff', 'health', 'health_max', 'hold_position', 'ideal_harvesters', 'is_blip', 'is_burrowed', 'is_enemy', 'is_flying', 'is_idle', 'is_mine', 'is_mineral_field', 'is_powered', 'is_ready', 'is_selected', 'is_snapshot', 'is_structure', 'is_vespene_geyser', 'is_visible', 'mineral_contents', 'move', 'name', 'noqueue', 'orders', 'owner_id', 'position', 'radar_range', 'radius', 'return_resource', 'shield', 'shield_max', 'stop', 'tag', 'train', 'type_id', 'vespene_contents', 'warp_in']
        '''
        self.expand_dis_dir = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            #print(distance_to_enemy_start)
            self.expand_dis_dir[distance_to_enemy_start] = el

        self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

        existing_ids = [unit.tag for unit in self.units]
        # removing of scouts that are actually dead now.
        to_be_removed = []
        for noted_scout in self.scouts_and_spots:
            if noted_scout not in existing_ids:
                to_be_removed.append(noted_scout)

        for scout in to_be_removed:
            del self.scouts_and_spots[scout]

        if len(self.units(ROBOTICSFACILITY).ready) == 0:
            unit_type = PROBE
            unit_limit = 1
        else:
            unit_type = OBSERVER
            unit_limit = 15

        assign_scout = True

        if unit_type == PROBE:
            for unit in self.units(PROBE):
                if unit.tag in self.scouts_and_spots:
                    assign_scout = False

        if assign_scout:
            if len(self.units(unit_type).idle) > 0:
                for obs in self.units(unit_type).idle[:unit_limit]:
                    if obs.tag not in self.scouts_and_spots:
                        for dist in self.ordered_exp_distances:
                            try:
                                location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                # DICT {UNIT_ID:LOCATION}
                                active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]

                                if location not in active_locations:
                                    if unit_type == PROBE:
                                        for unit in self.units(PROBE):
                                            if unit.tag in self.scouts_and_spots:
                                                continue

                                    await self.do(obs.move(location))
                                    self.scouts_and_spots[obs.tag] = location
                                    break
                            except Exception as e:
                                pass

        for obs in self.units(unit_type):
            if obs.tag in self.scouts_and_spots:
                if obs in [probe for probe in self.units(PROBE)]:
                    await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))


    async def intel(self):
        '''
        just simply iterate units.

        outline fighters in white possibly?

        draw pending units with more alpha

        '''

        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)


        for unit in self.units().ready:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (255, 255, 255), math.ceil(int(unit.radius*0.5)))


        for unit in self.known_enemy_units:
            pos = unit.position
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), int(unit.radius*8), (125, 125, 125), math.ceil(int(unit.radius*0.5)))

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

            plausible_supply = self.supply_cap / 200.0

            worker_weight = len(self.units(PROBE)) / (self.supply_cap-self.supply_left)
            if worker_weight > 1.0:
                worker_weight = 1.0

            cv2.line(game_data, (0, 19), (int(line_max*worker_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
            cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
            cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
            cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
            cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500
        except Exception as e:
            print(str(e))


        # flip horizontally to make our final fix in visual representation:
        grayed = cv2.cvtColor(game_data, cv2.COLOR_BGR2GRAY)
        self.flipped = cv2.flip(grayed, 0)
        #print(self.flipped)

        resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)


        if not HEADLESS:
            if self.use_model:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)
            else:
                cv2.imshow(str(self.title), resized)
                cv2.waitKey(1)

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    async def build_scout(self):
        for rf in self.units(ROBOTICSFACILITY).ready.noqueue:
            print(len(self.units(OBSERVER)), self.time/3)
            if self.can_afford(OBSERVER) and self.supply_left > 0:
                await self.do(rf.train(OBSERVER))
                break
        if len(self.units(ROBOTICSFACILITY)) == 0:
            pylon = self.units(PYLON).ready.noqueue.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                    await self.build(ROBOTICSFACILITY, near=pylon)


    async def build_worker(self):
        nexuses = self.units(NEXUS).ready.noqueue
        if nexuses.exists:
            if self.can_afford(PROBE):
                await self.do(random.choice(nexuses).train(PROBE))

    async def build_zealot(self):
        #if len(self.units(ZEALOT)) < (8 - self.time): # how we can phase out zealots over time?
        gateways = self.units(GATEWAY).ready.noqueue
        if gateways.exists:
            if self.can_afford(ZEALOT):
                await self.do(random.choice(gateways).train(ZEALOT))

    async def build_gateway(self):
        #if len(self.units(GATEWAY)) < 5:
        pylon = self.units(PYLON).ready.noqueue.random
        if self.can_afford(GATEWAY) and not self.already_pending(GATEWAY):
            await self.build(GATEWAY, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_voidray(self):
        stargates = self.units(STARGATE).ready.noqueue
        if stargates.exists:
            if self.can_afford(VOIDRAY):
                await self.do(random.choice(stargates).train(VOIDRAY))
        #####
        else:
            await self.build_stargate()

    async def build_stalker(self):
        pylon = self.units(PYLON).ready.noqueue.random
        gateways = self.units(GATEWAY).ready
        cybernetics_cores = self.units(CYBERNETICSCORE).ready

        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(STALKER):
                await self.do(random.choice(gateways).train(STALKER))

        if not cybernetics_cores.exists:
            if self.units(GATEWAY).ready.exists:
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_assimilator(self):
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

    async def build_stargate(self):
        cybernetics_cores = self.units(CYBERNETICSCORE)
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random
            if self.units(CYBERNETICSCORE).ready.exists:
                if self.can_afford(STARGATE) and not self.already_pending(STARGATE):
                    await self.build(STARGATE, near=pylon.position.towards(self.game_info.map_center, 5))

            ########################################
            if not cybernetics_cores.exists:
                if self.units(GATEWAY).ready.exists:
                    if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                        await self.build(CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_pylon(self):
            nexuses = self.units(NEXUS).ready
            if nexuses.exists:
                if self.can_afford(PYLON) and not self.already_pending(PYLON):
                    await self.build(PYLON, near=self.units(NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(NEXUS) and len(self.units(NEXUS)) < 3:
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.time + wait

    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.take_action(target))
            for u in self.units(STALKER).idle:
                await self.do(u.take_action(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.take_action(target))

    async def attack_known_enemy_structure(self):
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            for u in self.units(VOIDRAY).idle:
                await self.do(u.take_action(target))
            for u in self.units(STALKER).idle:
                await self.do(u.take_action(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.take_action(target))

    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
            for u in self.units(VOIDRAY).idle:
                await self.do(u.take_action(target))
            for u in self.units(STALKER).idle:
                await self.do(u.take_action(target))
            for u in self.units(ZEALOT).idle:
                await self.do(u.take_action(target))

    async def do_something(self):

        the_choices = {0: "build_scout",
                       1: "build_zealot",
                       2: "build_gateway",
                       3: "build_voidray",
                       4: "build_stalker",
                       5: "build_worker",
                       6: "build_assimilator",
                       7: "build_stargate",
                       8: "build_pylon",
                       9: "defend_nexus",
                       10: "attack_known_enemy_unit",
                       11: "attack_known_enemy_structure",
                       12: "expand",
                       13: "do_nothing"}

        if self.time > self.do_something_after:
            if self.use_model:
                worker_weight = 1
                zealot_weight = 1
                voidray_weight = 1
                stalker_weight = 1
                pylon_weight = 1
                stargate_weight = 1
                gateway_weight = 1
                assimilator_weight = 1

                prediction = self.model.predict([self.flipped.reshape([-1, 176, 200, 1])])
                weights = [1, zealot_weight, gateway_weight, voidray_weight, stalker_weight, worker_weight, assimilator_weight, stargate_weight, pylon_weight, 1, 1, 1, 1, 1]
                weighted_prediction = prediction[0]*weights
                choice = np.argmax(weighted_prediction)
                print('Choice:',the_choices[choice])
            else:
                worker_weight = 8
                zealot_weight = 3
                voidray_weight = 20
                stalker_weight = 8
                pylon_weight = 5
                stargate_weight = 5
                gateway_weight = 3

                choice_weights = 1*[0]+zealot_weight*[1]+gateway_weight*[2]+voidray_weight*[3]+stalker_weight*[4]+worker_weight*[5]+1*[6]+stargate_weight*[7]+pylon_weight*[8]+1*[9]+1*[10]+1*[11]+1*[12]+1*[13]
                choice = random.choice(choice_weights)

            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))

            y = np.zeros(14)
            y[choice] = 1
            self.train_data.append([y, self.flipped])

while True:
#if 1:
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, SentdeBot(use_model=True, title=1)),
        #Bot(Race.Protoss, SentdeBot(use_model=False, title=2)),
        Computer(Race.Protoss, Difficulty.Easy),
        ], realtime=False)