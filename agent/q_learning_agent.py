import os
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

    # Create folder for saving plots
    plot_dir = "plots"

    n_actions = env.action_space.n
    q_table = {}
    state_visits = defaultdict(int)
    rewards = []
    successes = []
    unique_states_per_episode = []

    sa_counts = defaultdict(int)

    # Heatmap trackers
    visit_counts = defaultdict(int)   # (x, y) -> count
    state_to_pos = {}                 # state -> (x, y)

    has_received_key_reward = False
    prev_q_table = {}

    for episode in trange(num_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0
        visited_states = set()

        # starting cell
        x0, y0 = env.unwrapped.agent_pos
        visit_counts[(x0, y0)] += 1
        state_to_pos[state] = (x0, y0)

        for step in range(max_steps):
            if render:
                env.render()

            if state not in q_table:
                q_table[state] = np.zeros(n_actions)

            state_visits[state] += 1
            visited_states.add(state)
            prev_agent_pos = env.unwrapped.agent_pos

            action = choose_action(state, q_table, prev_q_table, strategy, epsilon, n_actions, sa_counts)
            sa_counts[(state, action)] += 1
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

            # record next cell visits
            nx, ny = env.unwrapped.agent_pos
            visit_counts[(nx, ny)] += 1
            state_to_pos[next_state] = (nx, ny)

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
    plt.savefig(os.path.join(plot_dir, "total_reward_per_episode.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Rolling Success Rate
    window = 10
    rolling_success = [np.mean(successes[max(0, i - window):i + 1]) for i in range(num_episodes)]

    plt.figure()
    plt.plot(episodes, rolling_success)
    plt.title(f"Rolling Success Rate (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "rolling_success_rate.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Unique States Visited per Episode
    plt.figure()
    plt.plot(episodes, unique_states_per_episode)
    plt.title("Unique States Visited per Episode")
    plt.xlabel("Episode")
    plt.ylabel("# Unique States")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "unique_states_per_episode.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Epsilon Decay
    epsilons = [max(min_epsilon, config["epsilon"] * (epsilon_decay ** i)) for i in episodes]

    plt.figure()
    plt.plot(episodes, epsilons)
    plt.title("Epsilon Decay Over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "epsilon_decay.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Heatmaps
    if visit_counts:
        xs = [p[0] for p in visit_counts.keys()]
        ys = [p[1] for p in visit_counts.keys()]
        width = max(xs) + 1
        height = max(ys) + 1

        # 5.1 State-visit heatmap
        visit_grid = np.zeros((height, width), dtype=float)
        for (x, y), c in visit_counts.items():
            visit_grid[y, x] = c
        visit_grid_log = np.log1p(visit_grid)

        plt.figure()
        plt.imshow(visit_grid_log, origin="lower", interpolation="nearest")
        plt.colorbar(label="log(1 + visits)")
        plt.title("State Visit Heatmap")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plot_dir, "state_visit_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 5.2 Max-Q heatmap
        q_grid = np.full((height, width), np.nan, dtype=float)
        for s, qvals in q_table.items():
            if s in state_to_pos:
                x, y = state_to_pos[s]
                q_grid[y, x] = np.max(qvals)

        q_masked = np.ma.array(q_grid, mask=np.isnan(q_grid))

        plt.figure()
        plt.imshow(q_masked, origin="lower", interpolation="nearest")
        plt.colorbar(label="max Q(s, a)")
        plt.title("Max-Q Heatmap")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plot_dir, "max_q_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return q_table, rewards
