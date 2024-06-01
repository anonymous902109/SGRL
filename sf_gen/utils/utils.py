import copy
import itertools
import json
import os
import random
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from paretoset import paretoset
from pfrl.replay_buffers import EpisodicReplayBuffer
from tqdm import tqdm

from sf_gen.models.trajectory import Trajectory


def seed_everything(seed):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)


def generate_paths_with_outcome(outcome, csv_path, env, bb_model, n_ep=1000, horizon=5, traj_type='forward'):
    ''' Generates a dataset of Trajectory objects where a failure happens
    :param csv_path: path to save the dataset
    :param env: gym gym_env
    :param bb_model: a model used for prediciting next action in the gym_env
    :param n_ep: number of episodes
    :param horizon: number of iteractions before the failure that are saved in the failure trajectory
    '''
    try:
        # load buffer
        buffer = EpisodicReplayBuffer(capacity=100000000)
        buffer.load(csv_path)
        episodes = buffer.sample_episodes(n_episodes=min(500, len(buffer.episodic_memory))) # TODO: make num facts a parameter
        # transform into traj class
        return combine_trajs(episodes, outcome)
    except FileNotFoundError:
        print('Generating facts for outcome = {}'.format(outcome.name))
        buffer = EpisodicReplayBuffer(capacity=10000000)

        for i in tqdm(range(n_ep)):
            obs, _ = env.reset(int(datetime.now().timestamp()*1000000))
            done = False
            p = []

            while not done:
                action = bb_model.predict(obs)
                p.append((copy.copy(obs), action, None, None, copy.deepcopy(env.get_env_state())))

                new_obs, rew, done, trunc, info = env.step(action)
                done = done or trunc

                if (outcome.explain_outcome(env, new_obs)):  # if outcome should be explained
                    if (len(p) >= horizon) or (traj_type == 'forward'):  # either long back trajectory or we're looking forward so it does not matter

                        p.append((copy.copy(new_obs), None, None, None, copy.deepcopy(env.get_env_state())))
                        for t in p[-(horizon+1):]:
                            buffer.append(*t)

                        buffer.stop_current_episode()
                        done = True  # stop current episode when it's written into facts

                obs = new_obs

        # save buffer
        buffer.save(csv_path)
        episodes = buffer.sample_episodes(n_episodes=len(buffer.episodic_memory))

        return combine_trajs(episodes, outcome)

def combine_trajs(episodes, outcome):
    # transform into traj class
    trajs = []
    for e_id, e in enumerate(episodes):
        t = Trajectory(e_id, outcome)
        for i in e:
            t.append(i['state'], i['action'], i['next_action'])

        t.outcome.true_action = t.actions[-1]
        t.set_end_state(t.states[-1])
        trajs.append(t)
    return trajs


# TODO: could go into dataset class
def transform_from_baseline_format(df, env, task_name):
    if task_name == 'gridworld':
        df['cf'] = df.apply(lambda x: env.create_state(int(x.Agent),
                                                       int(x.Monster),
                                                       [{pos: int(x['Tree{}'.format(pos)])}for pos in env.TREE_POS if x['Tree{}'.format(pos)]]),
                            axis=1)

    elif task_name == 'frozen_lake':
        df['cf'] = df.apply(lambda x: env.create_state(int(x.Agent),
                                                       int(x.Exit),
                                                       [x['Frozen{}'.format(i)] for i in range(1, 7)]), axis=1)
    elif task_name == 'farm.json':
        df['cf'] = df.apply(lambda x: [x[c] for c in df.columns if c not in ['Fact id', 'Action']], axis=1)

    return df


def transform_to_baseline_format(facts, env, bb_model, task_name, outcome_action):
    facts = list(itertools.chain(*facts))
    data = []
    if 'gridworld' in task_name:
        for f in facts:
            state = f.states[-1]
            state = list(state) + [bb_model.predict(state) != outcome_action]

            data.append(state)

        df = pd.DataFrame(data, columns=['Agent', 'Monster', 'Tree2', 'Tree7', 'Tree12', 'Tree17', 'Tree22', 'Action'],
                          dtype=float)
        # df = df.drop_duplicates()

        return df

    elif 'frozen_lake' in task_name:
        for f in facts:
            state = f.states[-1]
            state = list(state) + [bb_model.predict(state) != outcome_action]
            data.append(state)

        df = pd.DataFrame(data,
                          columns=['Agent', 'Exit', 'Frozen1', 'Frozen2', 'Frozen3', 'Frozen4', 'Frozen5', 'Frozen6', 'Action'],
                          dtype=float)
        df = df.drop_duplicates()

        return df
    elif 'farm' in task_name:
        for f in facts:
            state = f.states[-1]
            state = list(state) + [bb_model.predict(state) != outcome_action]
            data.append(state)

        df = pd.DataFrame(data,
                          columns=['day#int365', 'max#°C', 'mean#°C', 'min#°C', 'consecutive_dry#day', 'stage',
                                   'population#nb', 'size#cm', 'fruits_per_plant#nb', 'fruit_weight#g', 'Action'],
                          dtype=float)
        df = df.drop_duplicates()

        return df
    else:
        raise Exception('No task named {}'.format(task_name))


def get_pareto_cfs(cfs):
    cost_array = []
    for cf in cfs:
        cost_array.append(list(cf.reward_dict.values()))

    cost_array = np.array(cost_array)
    print(cost_array.shape)

    is_efficient = paretoset(cost_array, sense=["min"] * cost_array.shape[1])

    best_cfs = [cfs[i] for i in range(len(cfs)) if is_efficient[i]]

    return best_cfs

