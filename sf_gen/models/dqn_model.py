import torch
import gymnasium as gym
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.evaluation import evaluate_policy


class DQNModel:

    def __init__(self, env, model_path, training_timesteps):
        self.model_path = model_path
        self.env = env
        self.training_timesteps = training_timesteps

        self.model = self.load_model(model_path, env)

    def load_model(self, model_path, env):
        try:
            # try loading the model if already trained
            model = DQN.load(model_path)
            model.env = env
            model.policy.to('cpu')
            print('Loaded bb model')
        except FileNotFoundError:
            # train a new model
            print('Training bb model')
            model = DQN('MlpPolicy',
                        env,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=5e-4,
                        buffer_size=150000,
                        learning_starts=200,
                        batch_size=128,
                        gamma=0.9,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        exploration_fraction=0.5,
                        verbose=1)

            model.learn(total_timesteps=self.training_timesteps)
            model.save(model_path)

        return model

    def predict(self, x):
        ''' Predicts a deterministic action in state x '''
        action, _ = self.model.predict(x, deterministic=True)
        return action.item()

    def get_action_prob(self, x, a):
        ''' Returns softmax probabilities of taking action a in x '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)
        probs = torch.softmax(q_values, dim=-1).squeeze()

        return probs[a].item()

    def predict_proba(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)
        probs = torch.softmax(q_values, dim=-1).squeeze()

        return probs

    def get_Q_vals(self, x):
        ''' Returns a list of Q values for taking any action in x '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).unsqueeze(0)

        q_values = self.model.policy.q_net(x)

        return q_values.squeeze().tolist()

    def evaluate(self):
        ''' Evaluates learned policy in the environment '''
        avg_rew = evaluate_policy(self.model, self.env, n_eval_episodes=1000, deterministic=True)
        print('Average reward = {}'.format(avg_rew))
        return avg_rew