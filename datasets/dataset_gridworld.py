import pandas as pd

from sf_gen.utils.utils import transform_to_baseline_format


class GridworldDataset:

    def __init__(self, env, bb_model, facts, target_action, outcome_name, num_facts):
        self.TARGET_NAME = 'Action'
        self.continuous_feature_names = []
        self.categorical_feature_names = ['Agent', 'Monster', 'Tree2', 'Tree7', 'Tree12', 'Tree17', 'Tree22']

        self.columns = ['Agent', 'Monster', 'Tree2', 'Tree7', 'Tree12', 'Tree17', 'Tree22']
        self.env = env
        self.bb_model = bb_model
        self.facts = facts
        self.target_action = target_action
        self.task_name = 'gridworld'
        self.outcome_name = outcome_name

    def get_dataset(self):
        """
        Assumes target class is binary 0 1, and that 1 is the semi-factual class
        """
        self.dataset_path_baseline_format = 'datasets/{}/facts_baseline_format_{}.csv'.format(self.task_name, self.outcome_name)
        self.dataset_path_regular_format = 'datasets/{}/facts_{}.csv'.format(self.task_name, self.outcome_name)

        try:
            # if a dataframe is available in the format suitable for the baselines
            df = pd.read_csv(self.dataset_path_baseline_format)
            self.num_samples = len(df)
            return df
        except FileNotFoundError:
            # otherwise transform dataset from a facts dataset
            df_transformed = transform_to_baseline_format(self.facts, self.env, self.bb_model, self.task_name, self.target_action)
            df_transformed.to_csv(self.dataset_path_baseline_format, index=False)
            self.num_samples = len(df_transformed)
            return df_transformed

    def make_human_readable(self, df):
        return df

    def actionability_constraints(self):
        #### increasing means "increasing" probability of loan
        #### based on common sense actionable directions

        meta_action_data = {
            'Agent': {'actionable': True,
                    'min': 0,
                    'max': 25,
                    'can_increase': True,
                    'can_decrease': True},

            'Monster': {'actionable': False,
                         'min': 0,
                         'max': 25,
                         'can_increase': False,
                         'can_decrease': False},

            'Tree2': {'actionable': True,
                    'min': 0,
                    'max': 2,
                    'can_increase': False,
                    'can_decrease': False},

            'Tree7': {'actionable': True,
                       'min': 0,
                       'max': 2,
                       'can_increase': False,
                       'can_decrease': False},

            'Tree12': {'actionable': True,
                        'min': 0,
                        'max': 2,
                        'can_increase': False,
                        'can_decrease': False},

            'Tree17': {'actionable': True,
                     'min': 0,
                     'max': 2,
                     'can_increase': False,
                     'can_decrease': False},

            'Tree22': {'actionable': True,
                         'min': 0,
                         'max': 2,
                         'can_increase': False,
                         'can_decrease': False}
        }

        return meta_action_data


