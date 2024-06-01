class AbstractOutcome:

    def __init__(self, bb_model, target_action=None, true_action=None):
        self.true_action = true_action
        self.target_action = target_action

        self.bb_model = bb_model

    def cf_outcome(self, env, state):
        return True

    def explain_outcome(self, env, state=None):
        return None