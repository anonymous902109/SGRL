import copy
from datetime import datetime

import gymnasium as gym
import numpy as np

from envs.abs_env import AbstractEnv


class Gridworld(AbstractEnv):

    def __init__(self):
        super(Gridworld, self).__init__()

        self.world_dim = 5
        self.state_dim = 7

        self.chopping = 0
        self.max_chopping = 1

        self.step_pen = -1
        self.goal_rew = 10

        self.max_steps = 100
        self.steps = 0

        self.lows = np.array([0] * self.state_dim)
        self.highs = np.array([25, 25, 2, 2, 2, 2, 2])
        self.observation_space = gym.spaces.Box(self.lows, self.highs, shape=(self.state_dim, ))
        self.action_space = gym.spaces.Discrete(6)

        self.ACTIONS = {'RIGHT': 0, 'DOWN': 1, 'LEFT': 2, 'UP': 3, 'CHOP': 4, 'SHOOT': 5}
        self.OBJECTS = {'AGENT': 1, 'MONSTER': 2, 'TREE': 3, 'KILLED_MONSTER': -1}

        self.TREE_TYPES = {1: 1, 2: 1}   # indicates # steps needed to destroy tree
        self.TREE_REWS = {1: -1, 2: -5}  # penalty for destroying object
        self.FERTILITY = {2: 0.2, 7: 0.2, 12: 0.2, 17: 0.2, 22: 0.2}  # prob of regrowth each step
        self.TREE_POS_TYPES = {2: 1, 7: 2, 12: 2, 17: 2, 22: 1}  # types of trees at different positions
        self.TREE_POS = [2, 7, 12, 17, 22]

        # random generator controlling all stochasticity in the environment
        self.random_generator = np.random.default_rng(seed=int(datetime.now().timestamp()*10000))

    def step(self, action):
        if isinstance(action, str):
            action = self.ACTIONS[action]

        new_state, done, rew = self.get_new_state(self.state, action)

        self.state = new_state
        self.steps += 1

        self.is_done = done

        return new_state.flatten(), rew, done, done, {}

    def create_state(self, agent, monster, trees, chopping=0, chopped_trees=[], killed_monster=False):
        state = np.zeros((self.state_dim, ))
        state[0] = agent
        state[1] = monster

        if killed_monster:
            state[1] = self.OBJECTS['KILLED_MONSTER']

        for t in trees:
            t_pos, t_type = tuple(t.items())[0]
            tree_ind = self.TREE_POS.index(t_pos)
            if t_pos not in chopped_trees:
                state[2 + tree_ind] = t_type

        return state

    def get_new_state(self, state, action):
        agent, monster, trees = self.get_objects(state)

        facing_monster = self.facing_obstacle(agent, [monster], action)
        facing_tree = self.facing_obstacle(agent, [list(t.keys())[0] for t in trees], action)

        chopped_trees = []

        rew = self.step_pen

        if action == 0:  # MOVE
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if (agent + 1) % self.world_dim != 0:
                    agent += 1
        elif action == 1:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent + self.world_dim < self.world_dim * self.world_dim:
                    agent += self.world_dim
        elif action == 2:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent % self.world_dim != 0:
                    agent -= 1
        elif action == 3:
            self.chopping = 0
            if facing_monster or facing_tree:  # Agent's path is blocked, cannot move
                agent = agent
            else:
                if agent >= self.world_dim:
                    agent -= self.world_dim
        elif action == 4:  # CHOP
            near_trees = self.get_neighboring_trees(agent, trees)
            if len(near_trees):
                t_pos, t_type = tuple(near_trees[0].items())[0]  # start with first tree
                self.chopping += 1
                if self.chopping >= self.TREE_TYPES[t_type]:
                    chopped_trees.append(t_pos)
                    self.chopping = 0

                    rew = self.TREE_REWS[t_type]
                    if rew == max(self.TREE_REWS.values()):
                        self.failure = True  # if the most costly action has been made

        elif action == 5:  # SHOOT
            self.chopping = 0
            if (int(agent / self.world_dim) == int(monster / self.world_dim)) or (agent % self.world_dim == monster % self.world_dim):
                free = self.check_if_path_free(agent, monster, trees)
                if free:
                    new_array = self.create_state(agent, monster, trees, self.chopping, killed_monster=True)
                    return new_array, True, self.goal_rew

            rew = -5

        # regrow trees in the middle column
        new_trees = self.regrow(trees, agent, monster, chopped_trees)
        trees += new_trees

        new_state = self.create_state(agent, monster, trees, self.chopping, chopped_trees)

        return new_state, self.steps >= self.max_steps, rew

    def regrow(self, trees, agent, monster, chopped_trees):
        tree_occupied = [list(t.keys())[0] for t in trees if list(t.values())[0] != 0]
        free_squares = [s for s in self.TREE_POS if (s not in tree_occupied) and (s not in chopped_trees) and (s != agent) and (s != monster)]
        if len(free_squares) == 0:
            return []

        new_trees = []
        for i in free_squares:
            p = self.FERTILITY[i]
            regrow_i = self.random_generator.choice([0, 1], p=[1-p, p])
            if regrow_i == 1:
                tree_type = self.random_generator.choice([1, 2])
                new_trees.append({i: tree_type})

        return new_trees

    def get_neighboring_trees(self, agent, trees):
        nts = []
        for t in trees:
            t_pos, t_type = tuple(t.items())[0]
            if t_type in self.TREE_TYPES.keys():
                if self.next_to_obstacle(agent, t_pos):
                    nts.append(t)

        return nts

    def facing_obstacle(self, agent, obstacles, action):
        for o in obstacles:
            if ((agent + 1 == o) and ((agent + 1) % self.world_dim != 0) and  action == self.ACTIONS['RIGHT']) \
                    or (agent + self.world_dim == o and action == self.ACTIONS['DOWN']) \
                    or ((agent - 1 == o) and (agent % self.world_dim != 0) and action == self.ACTIONS['LEFT']) \
                    or (agent - self.world_dim == o and action == self.ACTIONS['UP']):
                return True

        return False

    def next_to_obstacle(self, agent, obstacle):
        if ((agent + 1 == obstacle) and ((agent + 1) % self.world_dim != 0)) \
                or (agent + self.world_dim == obstacle) \
                or ((agent - 1 == obstacle) and (agent % self.world_dim != 0)) \
                or (agent - self.world_dim == obstacle):
            return True

        return False

    def check_if_path_free(self, agent, monster, trees):
        if int(agent / self.world_dim) == int(monster / self.world_dim):
            for t in trees:
                t_pos, t_type = tuple(t.items())[0]
                if t_pos > min([agent, monster]) and t_pos < max([agent, monster]):
                    return False

        if (agent % self.world_dim == monster % self.world_dim):
            for t in trees:
                t_pos, t_type = tuple(t.items())[0]
                if t_pos % self.world_dim == monster % self.world_dim and t_pos > min([agent, monster]) and t_pos < max([agent, monster]):
                    return False

        return True

    def reset(self, seed=0):
        self.random_generator = np.random.default_rng(seed=int(datetime.now().timestamp()*10000))
        self.failure = False
        self.is_done = False

        monster = self.random_generator.integers(0, self.world_dim * self.world_dim - 1)
        agent = self.random_generator.integers(0, self.world_dim * self.world_dim - 1)

        while agent % 5 > 1:
            agent = self.random_generator.integers(0, self.world_dim * self.world_dim - 1)

        while monster % 5 < 3:
            monster = self.random_generator.integers(0, self.world_dim * self.world_dim - 1)

        tree_wall = np.array(self.TREE_POS)
        tree_pos = self.random_generator.uniform(0, 1, 5) > 0.5
        tree_pos = tree_wall[tree_pos]
        trees = []
        for t in tree_pos:
            tree_type = self.random_generator.choice([1, 2])
            trees.append({t: tree_type})

        self.chopping = 0

        self.state = self.create_state(agent, monster, trees, self.chopping)

        self.steps = 0
        
        return self.state.flatten(), None

    def close(self):
        pass

    def render(self):
        self.render_state(self.state)

    def render_state(self, state):
        if isinstance(state, list):
            state = np.array(state)

        agent, monster, trees = self.get_objects(state)

        rendering = '---------------'
        print('STATE = {}'.format(state))

        for i in range(self.world_dim * self.world_dim):
            if i % self.world_dim == 0:
                rendering += '\n'

            if i == agent:
                rendering += ' A '
            elif i == monster:
                rendering += ' M '
            else:
                tree_found = False
                for t in trees:
                    t_pos, t_type = tuple(t.items())[0]
                    if i == t_pos:
                        rendering += ' T{} '.format(t_type)
                        tree_found = True

                if not tree_found:
                    rendering += ' - '

        rendering += '\n'
        rendering += '---------------'
        print(rendering)

    def get_objects(self, x):
        x = np.array(x).squeeze()

        agent = x[0]
        monster = x[1]

        trees = []
        for i, t_pos in enumerate(self.TREE_POS):
            if x[2 + i] in self.TREE_TYPES.keys():
                trees.append({t_pos: int(x[2 + i])})

        return agent, monster, trees

    def realistic(self, x):
        agent, monster, trees = self.get_objects(x)

        if agent is None:
            return False

        total_trees = len(trees)

        t_pos = [list(i.keys())[0] for i in trees]
        t_types = [list(i.values())[0] for i in trees]

        for i, t in enumerate(t_pos):
            if t not in self.TREE_POS:
                return False

            if t == agent or t == monster:
                return False

        if agent == monster:
            return False

        if agent > 25:
            return False
        if monster > 25:
            return False
        if total_trees > 5:
            return False

        return True

    def actionable(self, x, fact):
        return True

    def get_actions(self, state):
        return np.arange(self.action_space.n)

    def set_state(self, state):
        self.state = copy.deepcopy(state)
        self.chopping = self.state[-1]

    def check_done(self, state):
        killed_monster = state[1] == -1

        if killed_monster:
            return True

        return False

    def equal_states(self, s1, s2):
        return sum(s1 != s2) == 0

    def writable_state(self, s):
        agent, monster, trees = self.get_objects(s)
        ws = 'Agent: {} Monster: {} Trees: {}'.format(agent, monster, trees)
        return ws

    def check_failure(self):
        return self.failure

    def get_env_state(self):
        return self.random_generator

    def action_distance(self, a, b):
        return a != b

    def set_stochastic_state(self, state, env_state):
        ''' Changes the environment's current state to x '''
        self.set_state(state)

        self.random_generator = env_state  # TODO: set random generator to the same values

    def set_nonstoch_state(self, state, env_state):
        self.set_stochastic_state(state, env_state)
        self.random_generator = np.random.default_rng(seed=int(datetime.now().timestamp() * 10e5))  # reset random generator
