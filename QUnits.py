import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sc2.bot_ai import BotAI
from sc2.unit import Unit, Point2
from sc2.units import Units


class QUnits(Units):
    GAMMA = 0.99

    REPLAY_MEMORY_SIZE = 10_000
    MIN_REPLAY_MEMORY_SIZE = 1
    replay_memory = list()

    MINIBATCH_SIZE = 64

    DIRECTORY_PATH = "Qmodels"

    reward_values = {"kill_enemy": 2.5,
                     "kill_struct": 1,
                     "stay_alive": 0.01,
                     "die": -3,
                     "win": 25,
                     "lose": -25}

    def __init__(self, bot, units, game_data):
        from JarexSc2 import JarexSc2
        super().__init__(units, game_data)
        self.bot: JarexSc2 = bot

        print(f"--- __init__ call on QUnits ---")

        self.action_choices = {0: self.defend,
                               1: self.attack_closest_enemy,
                               2: self.attack_closest_structure,
                               3: self.grouping,
                               4: self.do_nothing}

        self.learn_step = 0

        self.current_reward = 0.0
        self.last_reward = 0.0

        self.last_kill_score = 0
        self.current_kill_score = 0
        self.kill_score = 0
        self.max_kill_score = 1

        self.last_destruction_score = 0
        self.current_destruction_score = 0
        self.destruction_score = 0
        self.max_destruction_score = 1

        self.done = False
        self.current_state = self.get_null_state()
        self.last_state = self.get_null_state()
        if torch.cuda.is_available():
            self.current_state.cuda()
            self.last_state.cuda()

        self.current_action = torch.zeros([len(self.action_choices)], dtype=torch.float32)
        self.last_action = torch.zeros([len(self.action_choices)], dtype=torch.float32)

        self.current_task = self.do_nothing

        if not os.path.exists(self.DIRECTORY_PATH):
            os.makedirs(self.DIRECTORY_PATH)

        self.model_path = f"{self.DIRECTORY_PATH}/QUnits.pth"

        self.model = self.load_model(self.model_path)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

        # setting target model
        self._target_model = type(self.model)(tuple(self.current_state.size())[0], len(self.action_choices))
        self._target_model = self._target_model.cuda() if torch.cuda.is_available() else self._target_model

    def __del__(self):
        print(f"--- __del__ call on QUnits ---")
        # self.done = True
        # self.current_reward = self.reward_values["lose"]
        # self.update_memory(self.current_state, self.current_action, self.current_reward, self.get_state(), self.done)
        # self._update_epsilon()
        # self._update_target_model()
        # self.save_model()

    def on_end(self, game_result):
        from sc2.data import Result
        print(f"--- on_end call on QUnits ---")
        self.done = True

        if game_result == Result.Victory:
            self.current_reward = self.reward_values["win"]
        else:
            self.current_reward = self.reward_values["lose"]
        self.update_memory(self.current_state, self.current_action, self.current_reward, self.get_null_state(),
                           self.done)
        self._update_epsilon()
        self._update_target_model()
        self.save_model()

    def remove(self, unit: Unit) -> None:
        # print(f"--- remove call on QUnits for unit {unit.tag} ---")
        super().remove(unit)
        if self.current_reward is None:
            self.current_reward = self.reward_values["die"]
        else:
            self.current_reward += self.reward_values["die"]
        self.update_memory(self.current_state, self.current_action, self.current_reward, self.get_state(), self.done)

    def update_kill_score(self):
        self.current_kill_score = self.bot.state.score.killed_value_units
        self.max_kill_score = max(self.current_kill_score, self.max_kill_score)
        self.kill_score = (self.current_kill_score - self.last_kill_score) / self.max_kill_score

        if self.current_reward is None:
            self.current_reward = self.kill_score * self.reward_values["kill_enemy"]
        else:
            self.current_reward += self.kill_score * self.reward_values["kill_enemy"]

        self.last_kill_score = self.current_kill_score
        print(f"kill score : {self.kill_score}")

    def update_destruction_score(self):
        self.current_destruction_score = self.bot.state.score.killed_value_structures
        self.max_destruction_score = max(self.current_destruction_score, self.max_destruction_score)
        self.destruction_score = (self.current_destruction_score - self.last_destruction_score) / self.max_destruction_score

        if self.current_reward is None:
            self.current_reward = self.destruction_score * self.reward_values["kill_struct"]
        else:
            self.current_reward += self.destruction_score * self.reward_values["kill_struct"]

        self.last_destruction_score = self.current_destruction_score
        print(f"destruction score : {self.destruction_score}")

    def get_null_state(self):
        start_x = -1.0
        start_y = -1.0
        pos_x = -1.0
        pos_y = -1.0
        state = [start_x, start_y, pos_x, pos_y, 0, self.bot.iteration, self.current_kill_score,
                 self.current_destruction_score]
        state.extend(self.get_null_enemy_state())
        return torch.FloatTensor(np.array(state))

    def get_null_enemy_state(self):
        enemy_start_x = -1.0
        enemy_start_y = -1.0
        state = [enemy_start_x, enemy_start_y]
        state.extend(self.get_null_state_on_knows_enemy_units())
        state.extend(self.get_null_state_on_knows_enemy_structures())
        return state

    def get_null_state_on_knows_enemy_units(self):
        enemies = list()
        center_x = -1.0
        center_y = -1.0
        nb = len(enemies)
        state = [nb, center_x, center_y]
        return state

    def get_null_state_on_knows_enemy_structures(self):
        enemies = list()
        center_x = -1.0
        center_y = -1.0
        nb = len(enemies)
        state = [nb, center_x, center_y]
        return state

    def get_state(self):
        if not self.exists:
            return self.get_null_state()
        start_x = self.bot.start_location[0]
        start_y = self.bot.start_location[1]
        pos_x = self.center[0]
        pos_y = self.center[1]
        state = [start_x, start_y, pos_x, pos_y, self.amount, self.bot.iteration, self.current_kill_score,
                 self.current_destruction_score]
        state.extend(self.get_enemy_state())
        return torch.FloatTensor(np.array(state))

    def get_enemy_state(self):
        enemy_start_x = self.bot.enemy_start_locations[0][0]
        enemy_start_y = self.bot.enemy_start_locations[0][1]
        state = [enemy_start_x, enemy_start_y]
        state.extend(self.get_state_on_knows_enemy_units())
        state.extend(self.get_state_on_knows_enemy_structures())
        return state

    def get_state_on_knows_enemy_units(self):
        enemies = self.bot.known_enemy_units
        center_x = enemies.center[0] if enemies else -1.0
        center_y = enemies.center[1] if enemies else -1.0
        nb = enemies.amount
        state = [nb, center_x, center_y]
        return state

    def get_state_on_knows_enemy_structures(self):
        enemies = self.bot.known_enemy_structures
        center_x = enemies.center[0] if enemies else -1.0
        center_y = enemies.center[1] if enemies else -1.0
        nb = enemies.amount
        state = [nb, center_x, center_y]
        return state

    async def move_up(self):
        for unit in self:
            await self.bot.do(unit.move(self.center + Point2((0, -1))))

    async def move_down(self):
        for unit in self:
            await self.bot.do(unit.move(self.center + Point2((0, 1))))

    async def move_left(self):
        for unit in self:
            await self.bot.do(unit.move(self.center + Point2((-1, 0))))

    async def move_right(self):
        for unit in self:
            await self.bot.do(unit.move(self.center + Point2((1, 0))))

    async def defend(self):
        self.bot.defend(units=self)

    async def attack_closest_enemy(self):
        if self.bot.known_enemy_units.exists:
            for unit in self:
                await self.bot.do(unit.attack(self.bot.known_enemy_units.closest_to(self.center).position))

    async def attack_closest_structure(self):
        if self.bot.known_enemy_structures.exists:
            for unit in self:
                await self.bot.do(unit.attack(self.bot.known_enemy_structures.closest_to(self.center).position))

    async def grouping(self):
        for unit in self:
            await self.bot.do(unit.attack(self.center))

    async def do_nothing(self):
        pass

    async def take_action(self):
        self.update_kill_score()
        self.update_destruction_score()

        self.last_state = self.current_state
        self.current_state = self.get_state()

        self.current_state = self.current_state.cuda() if torch.cuda.is_available() else self.current_state

        self.last_reward = self.current_reward
        if self.current_reward is None:
            self.current_reward = self.reward_values["stay_alive"]

        action = torch.zeros([len(self.action_choices)], dtype=torch.float32)

        random_action = random.random() <= self.model.epsilon
        if random_action:
            action_idx = torch.randint(len(self.action_choices), torch.Size([]), dtype=torch.int)
        else:
            action_idx = self.model(self.current_state)[0].argmax().item()
        action[action_idx] = 1

        self.last_action = self.current_action
        self.current_action = action

        # self.current_action = self.current_action.cuda() if torch.cuda.is_available() else self.current_action

        print(f"QUnits action: {self.action_choices[int(action_idx)]}")

        await self.action_choices[int(action_idx)]()
        self.current_task = self.action_choices[int(action_idx)]

        self.update_memory(self.last_state, self.last_action, self.last_reward, self.current_state, self.done)
        self.learn()
        self.current_reward = None

    def update_memory(self, state, action, reward, next_state, done):
        # save transition to replay memory
        state = state.unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        done = torch.from_numpy(np.array([done], dtype=np.float32)).unsqueeze(0)
        self.replay_memory.append((state.cpu(), action.cpu(), reward, next_state.cpu(), done))

        # if replay memory is full, remove the oldest transition
        if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
            self.replay_memory.pop(0)

    def learn(self):
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return None

        # sample random minibatch
        minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.MINIBATCH_SIZE))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        next_state_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # extract Q-value
        q_values = torch.sum(self.model(state_batch) * action_batch, dim=1)

        # get output for the next state
        # next_output_batch = self.model(next_state_batch)
        next_output_batch = self._target_model(next_state_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        # good explanation at https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/
        next_q_values = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                        else reward_batch[i] + self.GAMMA * torch.max(next_output_batch[i])
                                        for i in range(len(minibatch))))

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.model.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        next_q_values = next_q_values.detach()

        # calculate loss
        loss = self.model.criterion(q_values, next_q_values)

        # do backward pass
        loss.backward()
        self.model.optimizer.step()
        self.learn_step += 1
        return loss.cpu().item() / self.MINIBATCH_SIZE

    def _update_epsilon(self):
        self.model.epsilon *= self.model.EPSILON_DECAY
        self.model.epsilon = max(self.model.epsilon, self.model.MIN_EPSILON)
        print(f"Qunits -> Current epsilon: {self.model.epsilon}")

    def save_model(self):
        torch.save(self.model, self.model_path)

    def load_model(self, path):
        if os.path.isfile(path):
            return torch.load(path)
        else:
            return QUnitsNetwork(input_chanels=tuple(self.current_state.size())[0], hm_actions=len(self.action_choices))

    def _update_target_model(self):
        self._target_model.load_state_dict(deepcopy(self.model.state_dict()))


class QUnitsNetwork(nn.Module):
    LR = 0.003
    epsilon = 0.9
    EPSILON_DECAY = 0.990
    MIN_EPSILON = 0.001

    def __init__(self, input_chanels, hm_actions):
        super(QUnitsNetwork, self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_chanels, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, hm_actions),
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.LR)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.classifier(x)
        return x
