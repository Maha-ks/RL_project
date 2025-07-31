import numpy as np
import random
from utils.state_utils import entropy

def choose_action(state, q_table, strategy, epsilon, state_visits, n_actions):
    q_values = q_table.get(state, np.zeros(n_actions))
    visit_count = state_visits[state]

    if strategy == "fixed":
        adaptive_epsilon = epsilon
    elif strategy == "novelty":
        adaptive_epsilon = epsilon / (1 + visit_count)
    elif strategy == "entropy":
        q_probs = np.exp(q_values) / np.sum(np.exp(q_values))
        adaptive_epsilon = min(1.0, entropy(q_probs))
    elif strategy == "count":
        adaptive_epsilon = epsilon
    else:
        adaptive_epsilon = epsilon

    if random.uniform(0, 1) < adaptive_epsilon:
        return random.randint(0, n_actions - 1)
    return np.argmax(q_values)
