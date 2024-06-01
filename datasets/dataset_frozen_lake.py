import pandas as pd

from sf_gen.utils.utils import transform_to_baseline_format


class FrozenLakeDataset:

    def __init__(self, env, bb_model, facts, target_action, outcome_name, num_facts):
        self.TARGET_NAME = 'Action'
        self.continuous_feature_names = []
        self.categorical_feature_names = ['Agent', 'Exit', 'Frozen1', 'Frozen2', 'Frozen3', 'Frozen4', 'Frozen5',
                                          'Frozen6']

        self.columns = ['Agent', 'Exit', 'Frozen1', 'Frozen2', 'Frozen3', 'Frozen4', 'Frozen5', 'Frozen6']

        self.env = env
        self.bb_model = bb_model
        self.facts = facts
        self.target_action = target_action
        self.task_name = 'frozen_lake'
        self.outcome_name = outcome_name

        self.dataset_path_baseline_format = 'datasets/frozen_lake/facts_baseline_format_{}.csv'.format(
            self.outcome_name)
        self.dataset_path_regular_format = 'datasets/frozen_lake/facts_{}.csv'.format(self.outcome_name)

    def get_dataset(self):
        """
        Assumes target class is binary 0 1, and that 1 is the semi-factual class
        """
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
                      'can_increase': False,
                      'can_decrease': False},

            'Exit': {'actionable': True,
                     'min': 0,
                     'max': 25,
                     'can_increase': False,
                     'can_decrease': False},

            'Frozen1': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False},

            'Frozen2': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False},

            'Frozen3': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False},

            'Frozen4': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False},

            'Frozen5': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False},

            'Frozen6': {'actionable': False,
                        'min': 0,
                        'max': 25,
                        'can_increase': False,
                        'can_decrease': False}
        }

        return meta_action_data


