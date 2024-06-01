import copy

import numpy as np
from pfrl.replay_buffers import EpisodicReplayBuffer
from sklearn import preprocessing as pre

from sf_gen.utils.utils import combine_trajs, seed_everything


class HighlightDiv:

    def __init__(self, env, bb_model, num_states=10):
        self.env = env
        self.bb_model = bb_model
        self.num_states = num_states

    def generate_important_states(self, horizon, outcome):
        seed_everything(1)
        summary_importances = []
        summary_states = []
        summary_actions = []

        cummulative_reward = 0
        cummulative_steps = 0

        num_simulations = 100
        runs = 0

        buffer = EpisodicReplayBuffer(capacity=10000000)

        while runs < num_simulations:
            obs, _ = self.env.reset()
            done = False
            steps = 0

            p = []

            while not done:
                action = self.bb_model.predict(obs)
                Q_vals = self.bb_model.get_Q_vals(obs)

                p.append((copy.copy(obs), action, None, None, copy.deepcopy(self.env.get_env_state())))

                # compute importance
                importance = max(Q_vals) - min(Q_vals)

                new_obs, reward, done, trunc, info = self.env.step(action)

                # check if frame should be added to summary
                if (len(p) >= horizon) and not done:
                    # add frame to summary
                    summary_states.append(copy.copy(obs))
                    summary_importances.append(importance)
                    summary_actions.append(action)

                    # p.append((copy.copy(new_obs), None, None, None, copy.deepcopy(self.env.get_env_state())))
                    for t in p[-(horizon + 1):]:
                        buffer.append(*t)

                    buffer.stop_current_episode()

                    done = True

                steps += 1
                cummulative_steps += 1

                obs = new_obs

                cummulative_reward += reward

                if done:
                    break

            runs += 1

        max_indices = sorted(range(len(summary_importances)), key=lambda k: summary_importances[k], reverse=True)[:self.num_states]
        episodes = [e for i, e in enumerate(buffer.episodic_memory) if i in max_indices]

        self.save_summary(summary_states, summary_importances, summary_actions)
        print('---------------------------------------------------------------------------')

        return combine_trajs(episodes, outcome)

    def most_similar_state(self, state, added_states):
        differences = []
        q_vals_diffs = []
        state_diffs = []

        for s in added_states:
            q_vals_diff = sum(abs(np.subtract(np.array(self.bb_model.get_Q_vals(s)), np.array(self.bb_model.get_Q_vals(state)))))
            state_diff = sum(s != state)
            state_diffs.append([state_diff])
            q_vals_diffs.append([q_vals_diff])

        state_diffs = pre.MinMaxScaler().fit_transform(np.array(state_diffs))
        q_vals_diffs = pre.MinMaxScaler().fit_transform(np.array(q_vals_diffs))

        state_diffs = list(state_diffs.squeeze())
        # q_vals_diffs = list(q_vals_diffs.squeeze())

        differences = [state_diffs[i] for i in range(len(state_diffs))]

        min_diff_index = np.argmin(differences)

        return min_diff_index

    def save_summary(self, summary_states, summary_importances, summary_actions):

        for i, state in enumerate(summary_states):
            print('{} Action = {} Importance = {}'.format(self.env.writable_state(state), summary_actions[i], summary_importances[i]))
