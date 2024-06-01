import copy
import os
import ast

from tqdm import tqdm

from sf_gen.approaches.scf_expl import SemiCounterfactualGen
from sf_gen.approaches.spf_expl import SemiPrefactualGen
import pandas as pd
import numpy as np

from sf_gen.outcomes.exact_state_outcome import ExactStateOutcome
from sf_gen.utils.utils import get_pareto_cfs, transform_from_baseline_format


def evaluate_realistic_instances(result_dir, rules):
    print('Evaluating realistic instance')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        df_path = os.path.join(result_dir, baseline_name)
        for i, r in enumerate(rules):
            satisfied = 0.0
            total = 0.0
            for outcome_name in [x[2] for x in os.walk(df_path)]:
                df_path = os.path.join(df_path, outcome_name[0])

                df = pd.read_csv(df_path, header=0)

                df['rule{}'.format(i)] = df.apply(lambda x: r(ast.literal_eval(x['SF'])), axis=1)
                satisfied += sum(df['rule{}'.format(i)])
                total += len(df)

            print('Rule {}: Satisfied = {}'.format(i, (satisfied/total)*100))


def evaluate_informativeness(result_dir):
    print('Evaluating informativeness')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        df_path = os.path.join(result_dir, baseline_name)
        informativness = []
        num_sfs = 0.0
        for outcome_name in [x[2] for x in os.walk(df_path)]:
            df_path = os.path.join(df_path, outcome_name[0])

            df = pd.read_csv(df_path, header=0)

            df['informative'] = df.apply(lambda x: not (ast.literal_eval(x['SF']) == ast.literal_eval(x['Fact'])), axis=1)

            informativness.append(sum(df['informative']))
            num_sfs += len(df)

        print('Informativeness = {}'.format((np.sum(informativness) / num_sfs)*100))

def evaluate_diversity(result_dir):
    printout = '{: ^20}'.format('Algorithm') + '|' + \
               '{: ^20}'.format('Feature diversity') + '|' + '\n'
    printout += '-' * ((20 + 1) * 4) + '\n'

    print('Evaluating feature diversity')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        df_path = os.path.join(result_dir, baseline_name)

        for outcome_name in [x[2] for x in os.walk(df_path)]:
            df_path = os.path.join(df_path, outcome_name[0])
            df = pd.read_csv(df_path, header=0)

            feature_div = evaluate_feature_diversity(df)

            printout += '{: ^20}'.format(baseline_name) + '|' + \
                        '{: ^20.4}'.format(feature_div) + '|' + '\n'

    print(printout)

def evaluate_quantity(df):
    facts = pd.unique(df['Fact id'])

    cfs = []
    for f in facts:
        n = len(df[df['Fact id'] == f])
        cfs.append(n)

    return np.mean(cfs)

def evaluate_metric_diversity(df, param, obj):
    facts = pd.unique(df['Fact id'])
    metrics = obj.objectives
    diversity = []

    for f in facts:
        df_fact = df[df['Fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = 0
                    for m in metrics:
                        diff += (x[m] - y[m]) ** 2

                    diversity.append(diff)

    avg_div = np.mean(diversity)

    return avg_div


def evaluate_feature_diversity(df):
    facts = pd.unique(df['Fact id'])
    diversity = []

    for f in facts:
        df_fact = df[df['Fact id'] == f]
        for i, x in df_fact.iterrows():
            for j, y in df_fact.iterrows():
                if i != j:
                    diff = mse(np.array(ast.literal_eval(x['SF'])), np.array(ast.literal_eval(y['SF'])))
                    diversity.append(diff)

    avg_div = np.mean(diversity)
    return avg_div


def evaluate_generated_cfs(num_facts, result_dir):
    print('Evaluating number generated sfs')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        df_path = os.path.join(result_dir, baseline_name)
        generated = []
        num_outcomes = len([x[2] for x in os.walk(df_path)])
        for i, outcome_name in enumerate([x[2] for x in os.walk(df_path)]):
            df_path = os.path.join(df_path, outcome_name[0])

            df = pd.read_csv(df_path, header=0)

            unique_facts = df['Fact id'].unique().tolist()
            generated.append(len(unique_facts))

        generate_perc = (sum(generated) / (num_facts*num_outcomes)) * 100
        print('{}: Percentage generated sfs for {} facts = {}'.format(baseline_name, num_facts, generate_perc))


def evaluate_sf_properties(result_dir, obj):
    print('Evaluating Feature similarity')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        df_path = os.path.join(result_dir, baseline_name)
        results = {}
        for outcome_name in [x[2] for x in os.walk(df_path)]:
            df_path = os.path.join(df_path, outcome_name[0])
            df = pd.read_csv(df_path, header=0)

            obj_cons = list(set(obj.objectives + obj.constraints))
            res = []
            for m in obj_cons:
                val = np.mean(df[m])

                if m in results.keys():
                    results[m].append(val)
                else:
                    results[m] = [val]

            print('Results for {} {}'.format(baseline_name, outcome_name))
            print([{k: np.mean(v) for k, v in results.items()}])


def evaluate_feature_similarity(result_dir):
    # calculates feature similarity between factual and semi-factual states
    print('Evaluating Feature similarity')
    for baseline_name in os.listdir(result_dir):
        print('Algorithm = {}'.format(baseline_name))
        results = []
        total = 0.0
        df_path = os.path.join(result_dir, baseline_name)
        for outcome_name in [x[2] for x in os.walk(df_path)]:
            df_path = os.path.join(df_path, outcome_name[0])
            df = pd.read_csv(df_path, header=0)

            df['diff'] = df.apply(lambda x: mse(np.array(ast.literal_eval(x['SF'])), np.array(ast.literal_eval(x['Fact']))), axis=1)
            diff = sum(df['diff'])

            results.append(diff)
            total += len(df)

        print('Feature similarity for {} = {}'.format(baseline_name, diff / total))


def mse(x, y):
    return np.sqrt(sum(np.square(x - y)))

def evaluate_robustness(result_dir, bb_model):
    # TODO: evaluate robustness
    pass

def transform_baseline_results(env, bb_model, params, transition_model, facts, method_names, task_name, outcome_name, diversities):
    baseline_paths = ['results/{}/{}/{}_{}_readable.csv'.format(task_name, baseline_name, div, outcome_name) for baseline_name in method_names for div in diversities]

    baselines = [pd.read_csv(baseline_path) for baseline_path in baseline_paths]
    data = []

    for m_i, res in enumerate(baselines):
        print('Evaluating {}'.format(method_names[m_i]))
        res = transform_from_baseline_format(res, env, task_name)

        scf = SemiCounterfactualGen(env, bb_model, params, transition_model)
        spf = SemiPrefactualGen(env, bb_model, params, transition_model)

        eval_path = baseline_paths[m_i]

        for i, row in tqdm(res.iterrows()):
            fact_id = row['fact_id']
            f = facts[fact_id]
            cf = row['cf']

            # search for counterfactuals using these methods
            outcome = ExactStateOutcome(cf)
            fact = copy.deepcopy(f)
            fact.outcome = outcome

            obj_cons = list(set(scf.obj.objectives + scf.obj.constraints))

            # search for a semifactual in the neighborhood
            cfs = []
            cfs_forward = scf.generate_counterfactuals(fact)
            cfs_backward = spf.generate_counterfactuals(fact)
            cfs = cfs_backward + cfs_forward

            # no cfs found in the neighborhood
            if len(cfs) == 0:
                data = append_data(data, fact_id, fact, cf, None, obj_cons, env)

            else:
                 # select the best ones
                best_cfs = get_pareto_cfs(cfs)

                # write down results
                data = append_data(data, fact_id, fact, cf, best_cfs, obj_cons, env)

            write_results(data, eval_path, obj_cons)


def append_data(data, fact_id, fact, sf, sfs_paths, obj_cons, env):
    if sfs_paths is None:
        data.append((fact_id,
                     list(fact.end_state),
                     list(sf),
                     env.writable_state(fact.end_state),
                     env.writable_state(sf),
                     list(fact.actions),
                     np.nan,
                     *[1 for obj_name in obj_cons],
                     0,
                     0))
    else:
        for cf in sfs_paths:
            data.append((fact_id,
                         list(fact.end_state),
                         list(cf.cf),
                         env.writable_state(fact.end_state),
                         env.writable_state(cf.cf),
                         list(fact.actions),
                         list(cf.recourse),
                         *[cf.reward_dict[obj_name] for obj_name in obj_cons],
                         cf.value,
                         1))

    return data

def write_results(data, eval_path, obj_cons):
    columns = ['Fact id',
               'Fact',
               'SF',
               'Fact_readable',
               'SF_readable',
               'Fact actions',
               'Recourse'] + obj_cons + ['Value', 'Found']

    df = pd.DataFrame(data, columns=columns)

    df.to_csv(eval_path, index=False)