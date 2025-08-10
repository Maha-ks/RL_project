import numpy as np
import random
from utils.state_utils import entropy

def choose_action(state, q_table, prev_q_table, strategy, epsilon, n_actions, sa_counts=None, c=1.0, novelty_weights=None):
    q_values = q_table.get(state, np.zeros(n_actions))
    prev_q_values = prev_q_table.get(state, np.zeros(n_actions))

    if strategy == "decay":
        adaptive_epsilon = epsilon

    elif strategy == "novelty":
        # Learning progress
        learning_progress = np.max(np.abs(q_values - prev_q_values))

        # Entropy
        q_probs = softmax(q_values) if np.any(q_values) else np.ones(n_actions)/n_actions
        normalized_entropy = entropy(q_probs) / np.log(n_actions)

        # Q-value variance
        q_var = np.var(q_values)
        q_range = (np.max(q_values) - np.min(q_values)) ** 2 + 1e-8
        normalized_variance = q_var / q_range

        # Combine all into a normalized novelty score in [0, 1]
        '''novelty_score = (
            0.3 * normalized_entropy +
            0.4 * learning_progress +
            0.3 * normalized_variance
        )'''
        weights = locals().get("novelty_weights", {"entropy": 0.3, "progress": 0.4, "variance": 0.3})
        novelty_score = (
            weights["entropy"]  * normalized_entropy +
            weights["progress"] * learning_progress +
            weights["variance"] * normalized_variance
        )

        adaptive_epsilon = max(epsilon, novelty_score)
    elif strategy == "entropy":
        q_probs = softmax(q_values) if np.any(q_values) else np.ones(n_actions) / n_actions
        H = entropy(q_probs)
        H_norm = H / (np.log(n_actions) + 1e-12)
        adaptive_epsilon = max(epsilon, H_norm)

    elif strategy == "count":
        counts = np.array([sa_counts.get((state, a), 0) for a in range(n_actions)], dtype=float)
        bonus = c / np.sqrt(counts + 1.0)
        scores = q_values + bonus
        return int(np.argmax(scores))
    
    else:
        adaptive_epsilon = epsilon

    if random.uniform(0, 1) < adaptive_epsilon:
        return random.randint(0, n_actions - 1)
    return np.argmax(q_values)

def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / (e.sum() + 1e-8)