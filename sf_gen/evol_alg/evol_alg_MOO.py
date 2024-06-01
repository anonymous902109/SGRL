from sf_gen.evol_alg.MOOProblem import MOOProblem
from sf_gen.evol_alg.evol_alg import EvolutionaryAlg


class EvolutionAlgMOO(EvolutionaryAlg):

    def __init__(self, env, bb_model, obj, params):
        super(EvolutionAlgMOO, self).__init__(env, bb_model, obj, params)

    def generate_problem(self, fact, allow_noop=False):
        n_objectives = len(self.obj.objectives)
        n_constraints = len(self.obj.constraints)

        if allow_noop:
            self.xl = -1

        return MOOProblem(self.horizon, n_objectives, n_constraints, self.xl, self.xu, fact, self.obj)


