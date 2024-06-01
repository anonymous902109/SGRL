import pandas as pd
import numpy as np

from baselines.algorithms import genetic_algorithm, piece_algorithm, neme_algorithm


class Baselines():
    ''' Runs all baselines for SF generation'''
    def __init__(self, bb_model, diversity_sizes, task_name):
        self.DIVERSITY_SIZES = diversity_sizes
        self.POP_SIZES = [100] * len(self.DIVERSITY_SIZES)
        self.bb_model = bb_model
        self.task_name = task_name

    def generate_cfs(self, fact_dataset, results_path, test_n):

        for idx, DIVERSITY_SIZE in enumerate(self.DIVERSITY_SIZES):
            print("Diversity Size:", DIVERSITY_SIZE)

            POP_SIZE = self.POP_SIZES[idx]

            print('Running SGEN')
            genetic_algorithm(self.bb_model, DIVERSITY_SIZE, POP_SIZE, fact_dataset, results_path, test_size=test_n)
            print('Finished SGEN')

                # print('Running neme alg with seed = {}'.format(seed))
                # neme_algorithm(self.bb_model, DIVERSITY_SIZE, fact_dataset, results_path)
                # print('Finished neme alg with seed = {}'.format(seed))
                #
                # if DIVERSITY_SIZE == 1:
                #     print('Running piece alg with seed = {}'.format(seed))
                #     piece_algorithm(self.bb_model, DIVERSITY_SIZE, fact_dataset, results_path)
                #     print('Finished piece alg with seed = {}'.format(seed))

            print('Finished diversity = {}'.format(DIVERSITY_SIZE))