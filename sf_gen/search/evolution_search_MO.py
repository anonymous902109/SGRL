from sf_gen.evol_alg.evol_alg_MOO import EvolutionAlgMOO
from sf_gen.search.abstract_search import AbstractSearch


class EvolutionSearchMOO(AbstractSearch):

    def __init__(self, env, bb_model, obj, params):
        super(EvolutionSearchMOO, self).__init__(env, bb_model, obj, params)
        self.alg = EvolutionAlgMOO(env, bb_model, obj, params)

