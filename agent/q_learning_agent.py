from utils.state_utils import get_state
from .policies import choose_action
from utils.reward_shaping import shape_reward
from tqdm import trange
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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
    successes = []
    unique_states_per_episode = []

    has_received_key_reward = False
    prev_q_table = {}

    for episode in trange(num_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0
        visited_states = set()

        for step in range(max_steps):
            if render:
                env.render()

            if state not in q_table:
                q_table[state] = np.zeros(n_actions)

            state_visits[state] += 1
            visited_states.add(state)
            prev_agent_pos = env.unwrapped.agent_pos

            action = choose_action(state, q_table, prev_q_table, strategy, epsilon, n_actions)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = get_state(env, next_obs)

            old_q_values = q_table[state].copy()

            if next_state not in q_table:
                q_table[next_state] = np.zeros(n_actions)

            reward, has_received_key_reward = shape_reward(
                env, action, reward, strategy, state_visits,
                next_state, beta, has_received_key_reward, prev_agent_pos
            )

            prev_q_table = q_table.copy()
            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                alpha * (reward + gamma * np.max(q_table[next_state]))

            prev_q_table[state] = old_q_values 

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)
        successes.append(1 if done else 0)
        unique_states_per_episode.append(len(visited_states))

    env.close()

    episodes = list(range(num_episodes))

    # 1. Total Reward per Episode
    plt.figure()
    plt.plot(episodes, rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # 2. Rolling Success Rate
    window = 10
    rolling_success = [np.mean(successes[max(0, i - window):i + 1]) for i in range(num_episodes)]

    plt.figure()
    plt.plot(episodes, rolling_success)
    plt.title(f"Rolling Success Rate (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.show()

    # 3. Unique States Visited per Episode
    plt.figure()
    plt.plot(episodes, unique_states_per_episode)
    plt.title("Unique States Visited per Episode")
    plt.xlabel("Episode")
    plt.ylabel("# Unique States")
    plt.grid(True)
    plt.show()

    # 4. Epsilon Decay
    epsilons = [max(min_epsilon, config["epsilon"] * (epsilon_decay ** i)) for i in episodes]

    plt.figure()
    plt.plot(episodes, epsilons)
    plt.title("Epsilon Decay Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.show()

    return q_table, rewards
