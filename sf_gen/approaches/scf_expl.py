

from sf_gen.objectives.scf_expl_obj import ScfExplObj
from sf_gen.search.evolution_search_MO import EvolutionSearchMOO


class SemiCounterfactualGen:

    def __init__(self, env, bb_model, params, transition_model):
        self.env = env
        self.obj = ScfExplObj(env, bb_model, params, transition_model)
        self.optim = EvolutionSearchMOO(env, bb_model, self.obj, params)

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        cfs = self.optim.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return cfs