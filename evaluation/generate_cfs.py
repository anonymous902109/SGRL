import os
import time

import pandas as pd
from tqdm import tqdm

def generate_counterfactuals(methods, method_names, facts, outcome, env, eval_path, params):
    ''' Generates counterfactual explanations for each passed failure trajectory using each model '''
    print('Generating counterfactuals for {} facts'.format(len(facts)))
    for i_m, m in enumerate(methods):
        start = time.time()
        record = []
        eval_path_results = os.path.join(eval_path, f'{method_names[i_m]}/{outcome.name}_results.csv')
        print('Method = {}'.format(method_names[i_m]))
        for i, t in tqdm(enumerate(facts)):
            res = m.generate_counterfactuals(t)

            obj_cons = list(set(m.obj.objectives + m.obj.constraints))

            for cf in res:
                record.append([t.id,
                               list(t.end_state),
                               list(cf.cf),
                               list(t.actions),
                               list(cf.recourse),
                               *[cf.reward_dict[obj_name] for obj_name in obj_cons],
                               cf.value])

            columns = ['Fact id',
                       'Fact',
                       'SF',
                       'Fact actions',
                       'Recourse'] + obj_cons + ['Value']
            df = pd.DataFrame(record, columns=columns)
            df.to_csv(eval_path_results)

        end = time.time()
        print('Method = {} Task = {} Average time for one counterfactual = {}'.format(method_names[i_m],
                                                                                      params['task_name'],
                                                                                      (end - start) / len(facts)))
