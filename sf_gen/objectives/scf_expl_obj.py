import copy
from datetime import datetime

from sf_gen.objectives.abstract_obj import AbstractObjective


class ScfExplObj(AbstractObjective):
    '''
    A set of objectives and constraints used for generating backward counterfactuals in ACTER algorithm
    The action proximity is defined for continuous actions
    '''
    def __init__(self, env, bb_model, params, transition_model):

        super(ScfExplObj, self).__init__(env, bb_model, params, transition_model)
        self.bb_model = bb_model
        self.env = env
        self.transition_model = transition_model
        self.objectives = ['uncertainty', 'fidelity', 'sparsity', 'exceptionality']  # TODO: there is probably a better name for this
        self.constraints = ['num_cfs']  # validity essentially

        self.n_sim = params['n_sim']

        self.NoOp = -1

    def evaluate(self, fact, actions):
        # process actions to remove NoOp actions
        actions, first_action_index = self.process_actions(actions)

        # if the action list is empty or invalid because NoOp is in the middle of the sequence
        if len(actions) == 0:
            return {'uncertainty': 1,
                    'fidelity': 1,
                    'sparsity': 1,
                    'exceptionality': 1
                    }, {'num_cfs': True}, []

        # evaluate properties
        stochasticity, fidelity, exceptionality, num_cfs, cfs = self.calculate_stochastic_rewards(fact, actions, self.bb_model, first_action_index=0)
        reachability = self.reachability(actions)

        for cf in cfs:
            cf[1].update({'sparsity': reachability, 'uncertainty': stochasticity})

        objectives = {'uncertainty': stochasticity,
                      'fidelity': fidelity,
                      'sparsity': reachability,
                      'exceptionality': exceptionality}

        constraints = {'num_cfs': num_cfs == 0}

        return objectives, constraints, cfs

    def get_first_state(self, fact, first_action_index):
        return copy.copy(fact.states[first_action_index]), copy.deepcopy(fact.env_states[first_action_index])

