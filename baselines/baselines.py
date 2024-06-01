import pandas as pd
import numpy as np

from baselines.algorithms import genetic_algorithm, piece_algorithm, neme_algorithm


class Baselines():
    ''' Runs all baselines for SF generation'''
    def __init__(self, bb_model, diversity_sizes, task_name):
        self.DIVERSITY_SIZES = diversity_sizes
        self.POP_SIZES = [24] * len(self.DIVERSITY_SIZES)
        self.bb_model = bb_model
        self.task_name = task_name

    def generate_cfs(self, fact_dataset, results_path, fact_ids=None, test_n=100):

        for idx, DIVERSITY_SIZE in enumerate(self.DIVERSITY_SIZES):
            print("Diversity Size:", DIVERSITY_SIZE)

            POP_SIZE = self.POP_SIZES[idx]
            genetic_algorithm(self.bb_model, DIVERSITY_SIZE, POP_SIZE, fact_dataset, results_path, test_size=test_n, fact_ids=fact_ids)

            print('Finished diversity = {}'.format(DIVERSITY_SIZE))