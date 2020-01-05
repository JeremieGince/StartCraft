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


class QUnit:
    GAMMA = 0.99

    REPLAY_MEMORY_SIZE = 1_000
    MIN_REPLAY_MEMORY_SIZE = 100
    replay_memory = list()

    MINIBATCH_SIZE = 32

    DIRECTORY_PATH = "Qmodels"

    reward_values = {"kill_enemy": 10,
                     "dmg_enemy": 5,
                     "stay_alive": 0.1,
                     "die": -10,
                     "lose": -25}

    def __init__(self, unit: Unit, q_group: Units, bot: BotAI):  # Il faut le changer en placeholder pour accelerer l'instanciation et facilité la remise des rewards
        self.unit = unit
        self.tag = None if unit is None else unit.tag
        self.q_group = q_group
        self.bot = bot
        self.enable = False if unit is None else True

        print(f"--- __init__ call on QUnit {self.tag} ---")

        self.action_choices = {0: self.move_up,
                               1: self.move_down,
                               2: self.move_left,
                               3: self.move_right,
                               4: self.attack_closest_enemy,
                               5: self.attack_closest_structure,
                               6: self.do_nothing}

        self.learn_step = 0

        self.current_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.current_state = self.get_state()
        self.last_state = self.get_state()
        if torch.cuda.is_available():
            self.current_state.cuda()
            self.last_state.cuda()

        self.current_action = torch.zeros([len(self.action_choices)], dtype=torch.float32)
        self.last_action = torch.zeros([len(self.action_choices)], dtype=torch.float32)

        if not os.path.exists(self.DIRECTORY_PATH):
            os.makedirs(self.DIRECTORY_PATH)

        self.model_path = f"{self.DIRECTORY_PATH}/QUnit_{self.unit.name}.pth"

        self.model = self.load_model(self.model_path)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model

        # setting target model
        self._target_model = type(self.model)(tuple(self.current_state.size())[0], len(self.action_choices))
        self._target_model = self._target_model.cuda() if torch.cuda.is_available() else self._target_model

    def __del__(self):
        print(f"--- __del__ call on QUnit {self.tag} ---")
        self.done = True
        self.current_reward = self.reward_values["die"]
        self.update_memory(self.current_state, self.current_action, self.current_reward, self.get_state(), self.done)
        self._update_epsilon()
        self._update_target_model()
        self.save_model()

    def set_unit(self, unit: Unit):
        print(f"--- set_unit call on QUnit {unit.tag} ---")
        self.unit = unit
        self.tag = unit.tag
        self.done = False
        self.enable = True

    def unset_unit(self):
        print(f"--- unset_unit call on QUnit {self.tag} ---")
        self.enable = False
        self.done = True
        self.current_reward = self.reward_values["die"]
        self.update_memory(self.current_state, self.current_action, self.current_reward, self.get_state(), self.done)
        self._update_epsilon()
        self._update_target_model()

    def get_state(self):
        pos_x = self.unit.position3d[0]
        pos_y = self.unit.position3d[1]
        pos_z = self.unit.position3d[2]
        state = [pos_x, pos_y, pos_z, self.unit.can_attack, self.unit.can_attack_air, self.unit.can_attack_ground,
                 self.unit.is_idle, self.unit.cloak, self.unit.detect_range, self.unit.health, self.unit.health_max,
                 self.unit.weapon_cooldown, self.unit.energy, self.unit.energy_max,
                 self.unit.shield, self.unit.shield_max, self.unit.is_blip]
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
        state.extend(self.get_enemies_in_attack_range())
        return state

    def get_enemies_in_attack_range(self):
        enemies_in_range = list()
        for e in self.bot.known_enemy_units:
            if self.unit.target_in_range(e):
                enemies_in_range.append(e)
        return [len(enemies_in_range)]

    def get_state_on_knows_enemy_structures(self):
        enemies = self.bot.known_enemy_structures
        center_x = enemies.center[0] if enemies else -1.0
        center_y = enemies.center[1] if enemies else -1.0
        nb = enemies.amount
        state = [nb, center_x, center_y]
        state.extend(self.get_enemies_struct_in_attack_range())
        return state

    def get_enemies_struct_in_attack_range(self):
        enemies_in_range = list()
        for e in self.bot.known_enemy_structures:
            if self.unit.target_in_range(e):
                enemies_in_range.append(e)
        return [len(enemies_in_range)]

    async def move_up(self):
        await self.bot.do(self.unit.move(self.unit.position + Point2((0, -1))))

    async def move_down(self):
        await self.bot.do(self.unit.move(self.unit.position + Point2((0, 1))))

    async def move_left(self):
        await self.bot.do(self.unit.move(self.unit.position + Point2((-1, 0))))

    async def move_right(self):
        await self.bot.do(self.unit.move(self.unit.position + Point2((1, 0))))

    async def attack_closest_enemy(self):
        if self.bot.known_enemy_units.exists:
            await self.bot.do(self.unit.attack(self.bot.known_enemy_units.closest_to(self.unit)))

    async def attack_closest_structure(self):
        if self.bot.known_enemy_structures.exists:
            await self.bot.do(self.unit.attack(self.bot.known_enemy_structures.closest_to(self.unit)))

    async def do_nothing(self):
        pass

    async def take_action(self):
        self.last_state = self.current_state
        self.current_state = self.get_state()

        self.current_state = self.current_state.cuda() if torch.cuda.is_available() else self.current_state

        self.last_reward = self.current_reward
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

        await self.action_choices[int(action_idx)]()
        self.update_memory(self.last_state, self.last_action, self.last_reward, self.current_state, self.done)
        self.learn()

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

    def save_model(self):
        torch.save(self.model, self.model_path)

    def load_model(self, path):
        if os.path.isfile(path):
            return torch.load(path)
        else:
            return QUnitNetwork(input_chanels=tuple(self.current_state.size())[0], hm_actions=len(self.action_choices))

    def _update_target_model(self):
        self._target_model.load_state_dict(deepcopy(self.model.state_dict()))


class QUnitNetwork(nn.Module):
    LR = 0.003
    epsilon = 0.9
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.001

    def __init__(self, input_chanels, hm_actions):
        super(QUnitNetwork, self).__init__()

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
