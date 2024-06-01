from sf_gen.outcomes.abstract_outcome import AbstractOutcome


class OneActionOutcome(AbstractOutcome):

    def __init__(self, bb_model, target_action=None, true_action=None):
        super(OneActionOutcome, self).__init__( bb_model, target_action, true_action)

        self.name = 'why not {}'.format(self.target_action)  # TODO: insert human-readable

    def cf_outcome(self, env, state):
        a = self.target_action == self.bb_model.predict(state)
        return a  # counterfactual where one specific action is required

    def sf_outcome(self, state):  # sf does not change the outcome to target
        return self.bb_model.predict(state) != self.target_action

    def explain_outcome(self, env, state=None):

        if (not env.is_done) and (self.bb_model.predict(state) != self.target_action):   # TODO: reformat true action to something more meaningful
            return True

        return False