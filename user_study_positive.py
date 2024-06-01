import copy
import itertools
import json
import os
import numpy as np

from baselines.baselines import Baselines
from datasets.dataset_farm import FarmDataset
from datasets.dataset_frozen_lake import FrozenLakeDataset
from datasets.dataset_gridworld import GridworldDataset
from envs.farm0 import Farm0
from envs.frozen_lake import FrozenLake
from envs.gridworld import Gridworld
from evaluation.generate_cfs import generate_counterfactuals
from sf_gen.approaches.scf_expl import SemiCounterfactualGen
from sf_gen.approaches.spf_expl import SemiPrefactualGen
from sf_gen.models.dqn_model import DQNModel
from sf_gen.models.monte_carlo import MonteCarloTransitionModel
from sf_gen.outcomes.one_action_outcome import OneActionOutcome

from sf_gen.utils import seed_everything, generate_paths_with_outcome


def main(task_name):
    print('TASK = {} '.format(task_name))
    seed_everything(seed=1)

    test_n = 100

    # define paths
    model_path = 'trained_models/{}'.format(task_name)
    param_file = 'params/{}.json'.format(task_name)
    facts_path = 'datasets/{}/'.format(task_name)
    transition_model_path = 'datasets/{}/transition_model.obj'.format(task_name)
    eval_path = f'results/{task_name}/'

    if task_name == 'gridworld':
        env = Gridworld()
        training_timesteps = int(3e5)
    elif task_name == 'frozen_lake':
        env = FrozenLake()
        training_timesteps = int(3e5)
    elif task_name == 'farm':
        env = Farm0()
        training_timesteps = int(3e5)

    # load bb model
    bb_model = DQNModel(env, model_path, training_timesteps)
    bb_model.evaluate()

    # load parameters
    with open(param_file, 'r') as f:
        params = json.load(f)

    # define target outcomes

    one_action_outcomes = [OneActionOutcome(bb_model, target_action=a) for a in range(env.action_space.n)]
    outcomes = one_action_outcomes

    facts = []
    # generating a set of facts with an outcome (only doing backward ones for here)
    for o in outcomes:
        path_facts = os.path.join(facts_path, o.name)
        fact_with_outcome = generate_paths_with_outcome(o,
                                                        path_facts,
                                                        env,
                                                        bb_model,
                                                        horizon=params['horizon'],
                                                        traj_type='backward')
        # filter unique trajectories
        filtered_facts = []
        for f in fact_with_outcome:
            if not list(f.end_state) in [list(ff.end_state) for ff in filtered_facts]:
                filtered_facts.append(f)

        print('Generated {} facts for outcome {}'.format(len(filtered_facts), o.name))
        facts.append(filtered_facts)

    # define baseline facts (same as the one above just in a different format)
    baseline_facts = []
    for o in outcomes:
        if task_name == 'gridworld':
            baseline_fact_dataset = GridworldDataset(env, bb_model, facts, o.target_action, o.name, len(facts))

        elif task_name == 'frozen_lake':
            baseline_fact_dataset = FrozenLakeDataset(env, bb_model, facts, o.target_action, o.name, len(facts))

        elif task_name == 'farm.json':
            baseline_fact_dataset = FarmDataset(env, bb_model, facts, o.target_action, o.name, len(facts))

        baseline_facts.append(baseline_fact_dataset)

    # define baseline approaches
    diversity_sizes = [1]
    baselines = Baselines(bb_model, diversity_sizes, task_name)
    baseline_names = ['SGEN_1']

    user_study_fact_ids = []

    # generate sfs using baselines
    for f in baseline_facts:
       baselines.generate_cfs(f, eval_path, fact_ids=user_study_fact_ids, test_n=test_n)

    # define transition model
    transition_model = MonteCarloTransitionModel(env, bb_model, transition_model_path)

    # define sf approaches
    scf = SemiCounterfactualGen(env, bb_model, params, transition_model)
    spf = SemiPrefactualGen(env, bb_model, params, transition_model)

    methods = [scf, spf]
    method_names = ['SCF', 'SPF']

    # run scf and spf on facts that baselines ran on
    all_facts_all_outcomes = list(itertools.chain(*facts))
    for i, f in enumerate(all_facts_all_outcomes):
        f.id = i


    for i, o in enumerate(outcomes):
        fact_ids = user_study_fact_ids
        test_facts = [test_fact for j, test_fact in enumerate(all_facts_all_outcomes) if j in fact_ids]   # filter only test instances
        generate_counterfactuals(methods, method_names, test_facts, outcomes[i], env, eval_path, params)


if __name__ == '__main__':
    tasks = ['gridworld', 'frozen_lake']
    for t in tasks:
        main(t)