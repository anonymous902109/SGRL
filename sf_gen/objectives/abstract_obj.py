import copy
import math
import random
from datetime import datetime

import numpy as np

import torch


class AbstractObjective:
    ''' Describes an objective function for counterfactual search '''

    def __init__(self, env, bb_model, params, transition_model):
        self.transition_model = transition_model
        self.env = env
        self.bb_model = bb_model

        self.n_sim = params['n_sim']
        self.max_actions = params['max_actions']

        self.noop = -1

    def process_actions(self, actions, allow_first_noop=True):
        # process a sequence of actions to remove NoOp from start and end
        first_real_action_index = 0
        while (first_real_action_index < len(actions)) and (actions[first_real_action_index] == self.noop):
            first_real_action_index += 1

        # this is specifically for prefactuals, where first actions cannot be noops
        if not allow_first_noop and first_real_action_index != 0:
            return [], 0

        # if all actions are NoOp
        if first_real_action_index >= len(actions):
            return [], 0

        last_real_action_index = len(actions) - 1
        while actions[last_real_action_index] == self.noop:
            last_real_action_index -= 1

        between_actions = actions[first_real_action_index: (last_real_action_index + 1)]

        # if NoOp actions are not on start or the end
        if self.noop in between_actions:
            return [], 0

        return between_actions, first_real_action_index

    def get_first_state(self, fact, first_action_index=0):
        return None, None

    def validity(self, outcome, obs):
        valid_outcome = outcome.sf_outcome(obs)
        # IMPORTANT: return 1 if the class has changed -- to be compatible with minimization used by NSGA
        return not valid_outcome

    def sparsity(self, fact, actions, first_action_index):
        num_actions = len(actions)
        fact_actions = fact.actions[first_action_index: (first_action_index + num_actions)]

        # diff actions + sum of all actions not taken
        diff_actions = sum(np.array(fact_actions) != np.array(actions))
        extra_actions = len(fact.actions) - num_actions
        return ((diff_actions + extra_actions) / len(fact.actions))

    def reachability(self, actions):

        if len(actions) == 0:
            return 1

        return len(actions) * 1.0 / self.max_actions

    def calculate_stochastic_rewards(self, fact, actions, bb_model, first_action_index):
        n_sim = self.n_sim
        cfs = []

        target_outcome = 0.0
        fidelities = []

        exceptionallities = []

        for s in range(n_sim):
            randomseed = int(datetime.now().timestamp())
            self.env.reset(randomseed)
            first_state = self.get_first_state(fact, first_action_index)
            self.env.set_nonstoch_state(*first_state)

            fid = 0.0
            exc = 0.0

            if len(actions) == 0:
                return 1, 1, 1, []

            done = False
            early_break = False

            obs = first_state[0]

            for a in actions:
                if done:
                    early_break = True
                    break

                # calculate fidelity
                prob = bb_model.get_action_prob(obs, a)
                fid += prob

                # step in the environment
                new_obs, rew, done, trunc, _ = self.env.step(a)

                # calculate exceptionality
                trans_prob = self.transition_model.get_probability(list(obs), a, list(new_obs))
                exc += trans_prob

                obs = new_obs

            if not early_break and not done:
                # check if validity is satisfied
                validity = self.validity(fact.outcome, obs)
                target_outcome += int(not validity)

                if validity == 0:
                    # since 0 indicates that validity is satisfied
                    cfs.append((list(copy.copy(obs)), {'fidelity': fid / len(actions), 'exceptionality': exc / len(actions)}))

                fidelities.append(fid / len(actions))

        # calculate stochasticity
        # if outcome is confirmed everytime that is a bad thing
        stochasticity = (target_outcome / n_sim)

        # calculate fidelity
        if len(fidelities):
            fidelity = sum(fidelities) / (len(fidelities) * 1.0)
        else:
            fidelity = 1

        # calculate mean exceptionallity
        if len(exceptionallities):
            exceptional = sum(exceptionallities) / len(exceptionallities)
        else:
            exceptional = 1

        # calculate validity
        validity = 1 - target_outcome / n_sim

        # 1 - fidelity because we want to minimize it
        return stochasticity, 1 - fidelity, validity, exceptional, cfs