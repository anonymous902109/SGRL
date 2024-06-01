from sf_gen.objectives.spf_expl_obj import SpfExplObj
from sf_gen.search.evolution_search_MO import EvolutionSearchMOO


class SemiPrefactualGen():

    def __init__(self, env, bb_model, params, transition_model):
        self.env = env
        self.obj = SpfExplObj(env, bb_model, params, transition_model)
        self.optim = EvolutionSearchMOO(env, bb_model, self.obj, params)

        super(SemiPrefactualGen, self).__init__()

    def generate_counterfactuals(self, fact, target=None):
        return self.get_best_cf(fact, target)

    def get_best_cf(self, fact, target=None):
        cfs = self.optim.alg.search(init_state=fact, fact=fact, target_action=target, allow_noop=True)

        return cfs