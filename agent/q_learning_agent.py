from utils.state_utils import get_state
from .policies import choose_action
from utils.reward_shaping import shape_reward
from tqdm import trange
import numpy as np
from collections import defaultdict

def train_q_learning(env, strategy, config):
    alpha = config["alpha"]
    gamma = config["gamma"]
    epsilon = config["epsilon"]
    min_epsilon = config["min_epsilon"]
    epsilon_decay = config["epsilon_decay"]
    beta = config["beta"]
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps"]
    render = config.get("render", False)

    n_actions = env.action_space.n
    q_table = {}
    state_visits = defaultdict(int)
    rewards = []

    for episode in trange(num_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0

        for step in range(max_steps):
            if render:
                env.render()

            if state not in q_table:
                q_table[state] = np.zeros(n_actions)

            state_visits[state] += 1
            action = choose_action(state, q_table, strategy, epsilon, state_visits, n_actions)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(env, next_obs)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            reward = shape_reward(env, action, reward, strategy, state_visits, next_state, beta)

            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                alpha * (reward + gamma * np.max(q_table[next_state]))

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

    env.close()
    return q_table, rewards
