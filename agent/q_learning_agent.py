import os
from utils.state_utils import get_state
from .policies import choose_action
from utils.reward_shaping import shape_reward
from tqdm import trange
import numpy as np
import numpy.ma as ma
import random
from collections import defaultdict
import matplotlib.pyplot as plt


def train_q_learning(env, strategy, config, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
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
    env_name = env.spec.id.lower()  # e.g., "minigrid-doorkey-5x5-v0"
    if "doorkey" in env_name:
        env_type = "doorkey"
    elif "unlock" in env_name:
        env_type = "unlock"
    else:
        env_type = "other"
    seed_label = f"seed_{seed}" if seed is not None else "no_seed"
    strategy_label = strategy.replace(" ", "_")

    plot_dir = os.path.join("plots", env_type, strategy_label, seed_label)
    os.makedirs(plot_dir, exist_ok=True)

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

            #action = choose_action(state, q_table, prev_q_table, strategy, epsilon, n_actions, sa_counts)
            action = choose_action(
                state, q_table, prev_q_table, strategy, epsilon, n_actions,
                sa_counts, c=None, novelty_weights=config.get("novelty_weights")
            )
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
    window = 50  # smoothing window
    rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    final_avg = np.mean(rewards[-100:])  # average of last 100 episodes

    plt.figure()
    plt.plot(rewards, label="Raw Reward", alpha=0.4)
    plt.plot(range(window - 1, len(rewards)), rolling_rewards, label=f"Smoothed (window={window})", color='red')
    plt.axhline(final_avg, color='green', linestyle='--', label=f"Final avg: {final_avg:.2f}")
    plt.title(f"Total Reward per Episode ({strategy})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"total_reward_per_episode.png"), dpi=300, bbox_inches="tight")
    plt.close()
    ### option 2
    plt.figure()
    plt.plot(episodes, rewards)
    plt.title(f"Total Reward per Episode ({strategy})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"total_reward_per_episode1.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Rolling Success Rate
    window = 10
    rolling_success = [np.mean(successes[max(0, i - window):i + 1]) for i in range(num_episodes)]

    plt.figure()
    plt.plot(episodes, rolling_success)
    plt.title(f"Rolling Success Rate (window={window}) - {strategy}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"rolling_success_rate.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Unique States Visited per Episode
    plt.figure()
    plt.plot(episodes, unique_states_per_episode)
    plt.title(f"Unique States Visited per Episode ({strategy})")
    plt.xlabel("Episode")
    plt.ylabel("# Unique States")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"unique_states_per_episode.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Epsilon Decay
    epsilons = [max(min_epsilon, config["epsilon"] * (epsilon_decay ** i)) for i in episodes]

    plt.figure()
    plt.plot(episodes, epsilons)
    plt.title(f"Epsilon Decay Over Episodes ({strategy})")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"epsilon_decay.png"), dpi=300, bbox_inches="tight")
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
        plt.title(f"State Visit Heatmap ({strategy})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plot_dir, f"state_visit_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # 5.2 Max-Q heatmap
        
        # Construct Q-value heatmap matrix
        q_matrix = np.zeros((height, width))  
        mask = np.ones_like(q_matrix, dtype=bool)

        for state, q_vals in q_table.items():
            if state in state_to_pos:
                x, y = state_to_pos[state]
                q_matrix[y, x] = np.max(q_vals)
                mask[y, x] = False  # mark as visited

        masked_q = ma.masked_array(q_matrix, mask)

        # Plot with masked color for unvisited
        plt.figure()
        cmap = plt.cm.viridis
        cmap.set_bad(color='lightgray')  # gray for walls/unvisited
        plt.imshow(masked_q, origin='lower', cmap=cmap)
        plt.colorbar(label="max Q(s, a)")
        plt.title(f"Max Q Heatmap ({strategy})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(plot_dir, f"max_q_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
        

    return q_table, rewards
