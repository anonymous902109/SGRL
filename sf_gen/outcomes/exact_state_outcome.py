from sf_gen.outcomes.abstract_outcome import AbstractOutcome


class ExactStateOutcome():

    def __init__(self, state):
        self.state = list(state)
        self.name = 'exact state = {}'.format(state)

    def sf_outcome(self, x):
        return (self.state == list(x))
