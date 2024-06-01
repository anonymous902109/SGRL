import copy
import itertools
import json
import os

import pandas as pd
import numpy as np

from baselines.baselines import Baselines
from datasets.dataset_gridworld import GridworldDataset
from envs.gridworld import Gridworld
from evaluation.generate_cfs import generate_counterfactuals
from sf_gen.approaches.scf_expl import SemiCounterfactualGen
from sf_gen.approaches.spf_expl import SemiPrefactualGen
from sf_gen.models.dqn_model import DQNModel
from sf_gen.models.monte_carlo import MonteCarloTransitionModel
from sf_gen.outcomes.one_action_outcome import OneActionOutcome
from sf_gen.utils.user_study_util import HighlightDiv
from sf_gen.utils.utils import seed_everything


def main(task_name):
    seed_everything(seed=1)

    test_n = 20

    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)
    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)
    facts_path = 'datasets/user_study_gridworld/{}/'.format(task_name)
    transition_model_path = 'datasets/{}/transition_model.obj'.format(task_name)
    eval_path = f'results/user_study/{task_name}/'

    env = Gridworld()
    training_timesteps = int(3e5)

    # load bb model
    bb_model = DQNModel(env, model_path, training_timesteps)
    bb_model.evaluate()

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)

    outcomes = [OneActionOutcome(bb_model, target_action=4), OneActionOutcome(bb_model, target_action=5)]

    summary_generator = HighlightDiv(env, bb_model, num_states=test_n)
    facts = []

    important_states = summary_generator.generate_important_states(params['horizon'], outcomes[0])
    for o in outcomes:
        f = copy.deepcopy(important_states)
        for fact in f:
            fact.outcome = copy.copy(o)

        facts.append(f)

    baseline_facts = []
    for o in outcomes:
        baseline_fact_dataset = GridworldDataset(env, bb_model, facts, o.target_action, o.name, len(facts))
        baseline_fact_dataset.task_name = 'user_study_gridworld'
        baseline_facts.append(baseline_fact_dataset)

    diversity_sizes = [1]
    baselines = Baselines(bb_model, diversity_sizes, task_name)
    baseline_names = ['SGEN_1']

    # generate sfs using baselines
    fact_ids = np.random.choice(np.arange(0, test_n), test_n)
    for f in baseline_facts:
        baselines.generate_cfs(f, eval_path, test_n=test_n, fact_ids=fact_ids)

    # define transition model
    transition_model = MonteCarloTransitionModel(env, bb_model, transition_model_path)

    # define sf approaches
    scf = SemiCounterfactualGen(env, bb_model, params, transition_model)
    spf = SemiPrefactualGen(env, bb_model, params, transition_model)

    methods = [spf, scf]
    method_names = ['SPF', 'SCF']

    # run scf and spf on facts that baselines ran on
    all_facts_all_outcomes = list(itertools.chain(*facts))
    for i, f in enumerate(all_facts_all_outcomes):
        f.id = i

    for i, o in enumerate(outcomes):
        fact_ids = np.load('data/user_study_gridworld/1/{}/fact_ids.npy'.format( o.name))
        test_facts = [test_fact for j, test_fact in enumerate(all_facts_all_outcomes) if j in fact_ids]
        for f in test_facts:
            f.outcome = copy.copy(o)  # filter only test instances
        generate_counterfactuals(methods, method_names, test_facts, outcomes[i], env, eval_path, params)

    outcome_names = ['why not 4', 'why not 5']
    method_names = ['SCF', 'SPF', 'SGEN_1']

    common_facts = set()
    for method in method_names:
        for o in outcome_names:
            df_path = os.path.join(eval_path, '{}/{}_results.csv'.format(method, o))
            df = pd.read_csv(df_path, header=0)

            unique_facts = df['Fact id'].unique()

            if len(common_facts) == 0:
                common_facts.update(unique_facts)
            else:
                common_facts = common_facts.intersection(unique_facts)

    training_facts = np.random.choice(list(common_facts), int(len(common_facts)/2), False)
    test_facts = np.random.choice([i for i in np.arange(0, test_n) if i not in training_facts],  int(len(common_facts)/2), False)

    for method in method_names:
        for o in outcome_names:
            print('{} {}'.format(method, o))
            df_path = os.path.join(eval_path, '{}/{}_results.csv'.format(method, o))
            df = pd.read_csv(df_path, header=0)
            for f in training_facts:
                fact_df = df[df['Fact id'] == f]
                random_sf_index = np.random.choice(fact_df.index)

                print('Fact = {} index = {}'.format(f, random_sf_index))

    print(test_facts)


if __name__ == '__main__':
    tasks = ['gridworld']

    main(tasks[0])