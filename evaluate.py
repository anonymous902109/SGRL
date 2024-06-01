
import itertools
import json
import os

from envs.frozen_lake import FrozenLake
from envs.gridworld import Gridworld
from evaluation.evaluate import transform_baseline_results, evaluate_informativeness, evaluate_realistic_instances, \
    evaluate_feature_similarity, evaluate_sf_properties, evaluate_diversity, evaluate_generated_cfs

from sf_gen.models.dqn_model import DQNModel
from sf_gen.models.monte_carlo import MonteCarloTransitionModel
from sf_gen.objectives.scf_expl_obj import ScfExplObj
from sf_gen.outcomes.one_action_outcome import OneActionOutcome
from sf_gen.utils.utils import seed_everything, generate_paths_with_outcome


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

    # define transition model
    transition_model = MonteCarloTransitionModel(env, bb_model, transition_model_path)

    # Evaluation
    results_dir = 'results/{}/'.format(task_name)
    evaluate_generated_cfs(test_n, results_dir)
    # evaluate_realistic_instances(results_dir, [env.realistic])
    evaluate_feature_similarity(results_dir)
    sf_obj = ScfExplObj(env, bb_model, params, transition_model)
    evaluate_sf_properties(results_dir, sf_obj)
    evaluate_diversity(results_dir, )



if __name__ == '__main__':
    tasks = ['frozen_lake']
    for t in tasks:
        main(t)