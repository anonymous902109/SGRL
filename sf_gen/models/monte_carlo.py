
import pickle
import random

from tqdm import tqdm


class MonteCarloTransitionModel:
    def __init__(self, env, bb_model, path):
        self.env = env
        self.bb_model = bb_model

        self.transition_model_path = path

        try:
            with open(self.transition_model_path, 'rb') as f:
                self.mc_tree = pickle.load(f)
        except FileNotFoundError:
            self.mc_tree = MCTree()
            self.simulate(n_sim=10000)

            with open(self.transition_model_path, 'wb') as f:
                pickle.dump(self.mc_tree, f)

    def simulate(self, n_sim):
        print('Simulating for {} episodes'.format(n_sim))
        for i in tqdm(range(n_sim)):
            done = False
            state, _ = self.env.reset()
            while not done:
                random_action = random.choices([0, 1], weights=[0.8, 0.2])[0]
                if random_action:
                    action = self.env.action_space.sample()
                else:
                    action = self.bb_model.predict(state)

                new_state, reward, done, trunc, _ = self.env.step(action)

                self.mc_tree.append(list(state), action, list(new_state))

                state = new_state

    def get_probability(self, state, action, new_state):
        if state in self.mc_tree.states:
            state_id = self.mc_tree.states.index(state)
            node = self.mc_tree.nodes[state_id]

            prob = node.get_probability(action, new_state)
            return prob
        else:
            return 1


class MCNode:

    def __init__(self, state):
        self.state = state
        self.children = {}
        self.visited = 1.0

    def add_child(self, action, node):
        if action not in self.children.keys():
            self.children[action] = []

        if node.state not in [child_node.state for child_node in self.children[action]]:
            self.children[action].append(node)
            node.visited = 1.0
        else:
            existing_node = [child_node for child_node in self.children[action] if child_node.state == node.state][0]
            existing_node.add_visit()

    def add_visit(self):
        self.visited += 1

    def get_probability(self, action, new_state):
        if action not in self.children.keys():
            return 1

        n_visited = self.visited
        child_nodes = [child_node for child_node in self.children[action] if child_node.state == new_state]
        if len(child_nodes) == 0:
            return 1

        child_node = child_nodes[0]
        child_visited = child_node.visited
        return child_visited / n_visited


class MCTree:

    def __init__(self):
        self.nodes = []
        self.states = []

    def append(self, state, action, new_state):
        if any(s == state for s in self.states):
            state_id = self.states.index(state)
            node = self.nodes[state_id]

            child_node = MCNode(new_state)
            node.add_child(action, child_node)

            node.add_visit()
        else:
            node = MCNode(state)
            self.states.append(state)
            self.nodes.append(node)