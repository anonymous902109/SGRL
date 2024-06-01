# import copy
# from datetime import datetime
#
# import numpy as np
#
# from envs.abs_env import AbstractEnv
# import gymnasium as gym
#
#
# class Farm0(AbstractEnv):
#
#     def __init__(self):
#
#         # self.gym_env = env_maker()
#         self.gym_env.reset()
#
#         self.state_dim = 10
#         self.action_dim = 3
#         self.observation_space = gym.spaces.Box(low=np.zeros((self.state_dim, )), high=np.array([1000] * self.state_dim), shape=(self.state_dim, ))
#         self.action_space = gym.spaces.Discrete(self.action_dim)
#
#         self.action_type = 'discrete'
#         self.action_space_low = 0
#         self.action_space_high = 2
#
#     def step(self, action):
#         self.is_done = False
#         action = self.decode_action(action)
#         try:
#             obs, rew, done, trunc, _ = self.gym_env.step(action)
#         except:
#             obs, rew, done, trunc, _ = self.gym_env.step(action)
#
#         done = done or trunc
#
#         flat_obs = self.flatten_obs(obs)
#
#         stage = self.gym_env.fields['Field-0'].entities['Plant-0'].variables['stage'].item().value
#         if stage == 'dead' or stage == 'none':
#             self.failure = True
#             done = True
#             self.is_done = True
#
#         self.state = np.array(flat_obs)
#
#         return self.state, rew, done, trunc,  {"composite_obs": obs}
#
#     def reset(self, seed=None):
#         self.is_done = False
#         if seed is None:
#             seed = int(datetime.now().timestamp())
#
#         self.failure = False
#
#         obs, info = self.gym_env.reset(seed)
#         flat_obs = self.flatten_obs(obs)
#
#         self.state = np.array(flat_obs)
#
#         return self.state, info
#
#     def render(self):
#         print(self.state)
#
#     def flatten_obs(self, obs):
#         # take only the first element of the tuple that contains the obs -- only after reset
#         obs = copy.copy(obs)
#         if isinstance(obs, tuple):
#             obs = obs[0]
#
#         flat_obs = []
#         # obs is a list of dictionaries
#         for d in obs:
#             while isinstance(d, dict):
#                 vals = list(d.values()) # there is only one key-value pair in all dicts
#                 d = vals[0]
#
#             if isinstance(vals, list):
#                 for e in vals:
#                     flat_obs.append(e)
#             else:
#                 flat_obs.append(vals)
#
#         return [e if not isinstance(e, list) else e[0] for e in flat_obs]  # flatten list
#
#     def decode_action(self, action):
#         return [action]
#
#     def get_actions(self, x):
#         """ Returns a list of actions available in state x"""
#         return np.arange(0, self.action_space.n)
#
#     def set_stochastic_state(self, state, env_state):
#         self.set_state(copy.deepcopy(env_state))
#         self.gym_env.np_random = env_state[0]
#
#     def set_nonstoch_state(self, state, env_state):
#         self.set_state(copy.deepcopy(env_state))
#
#         # reset random generators to allow stochasticity
#         self.gym_env.np_random = np.random.RandomState(int(datetime.now().timestamp()))
#         for e in self.gym_env.fields['Field-0'].entities:
#             self.gym_env.fields['Field-0'].entities[e].np_random = np.random.RandomState(int(datetime.now().timestamp()))
#
#     def set_state(self, x):
#         # TODO: do this for all params not just ones passed by obs
#         """ Changes the environment"s current state to x """
#         field = self.gym_env.fields['Field-0']
#
#         field.entities['Weather-0'] = x[1]['Weather-0']
#         field.entities['Soil-0'] = x[1]['Soil-0']
#         field.entities['Plant-0'] = x[1]['Plant-0']
#
#         self.gym_env.farmers['BasicFarmer-0'].fields['Field-0'] = field
#
#     def check_done(self, x):
#         """ Returns a boolean indicating if x is a terminal state in the environment"""
#         return False
#
#     def equal_states(self, x1, x2):
#         """ Returns a boolean indicating if x1 and x2 are the same state"""
#         return sum(x1 != x2) == 0
#
#     def writable_state(self, x):
#         """ Returns a string with all state information to be used for writing results"""
#         x = self.unflatten(x)
#         writable_state = ''.join(['{}: {} '.format(k, v) for k, v in x.items()])
#         return writable_state
#
#     def unflatten_obs(self, x):
#
#         x = {'day#int365': x[0],
#              'max#°C': x[1],
#              'mean#°C': x[2],
#              'min#°C': x[3],
#              'consecutive_dry#day': x[4],
#              'stage': x[5],
#              'population#nb': x[6],
#              'size#cm': x[7],
#              'fruits_per_plant#nb': x[8],
#              'fruit_weight#g': x[9]}
#
#         return x
#
#     def check_failure(self):
#         ''' Returns a boolean indicating if a failure occured in the environment'''
#         return self.failure
#
#     def get_env_state(self):
#         return (self.gym_env.np_random, self.gym_env.fields['Field-0'].entities)
#
#     def action_distance(self, a, b):
#         '''Calculates distance between 2 actions in the environment '''
#         if a != 10 and b != 10:  # both actions are not harvest
#             return abs(a - b) / 10
#         else:
#             return a != b
#
#
