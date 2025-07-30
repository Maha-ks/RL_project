# using empty env
import gymnasium as gym
import minigrid
import numpy as np
import random
from tqdm import trange

# Parameters
alpha = 0.1              # Learning rate
gamma = 0.99             # Discount factor
epsilon = 1.0            # Initial exploration rate
min_epsilon = 0.01       # Minimum exploration rate
epsilon_decay = 0.995    # Decay rate per episode
num_episodes = 5000      # Total training episodes
max_steps = 50          # Max steps per episode

# Environment
minigrid = "MiniGrid-Empty-5x5-v0"

env = gym.make(minigrid, render_mode=None)
obs_space_size = env.observation_space["image"].shape
n_actions = env.action_space.n

# Q-Table: Use flattened (x, y, direction) as state
q_table = {}

def get_state(obs):
    """Convert image observation into a hashable state"""
    return tuple(obs["image"].flatten())

def choose_action(state):
    """Epsilon-greedy action selection"""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table.get(state, np.zeros(n_actions)))

# Training
rewards = []

for episode in trange(num_episodes):
    obs, _ = env.reset()

    # Randomize agent position in training too
    while True:
        x = random.randint(0, env.unwrapped.width - 1)
        y = random.randint(0, env.unwrapped.height - 1)
        if env.unwrapped.grid.get(x, y) is None:
            env.unwrapped.agent_pos = (x, y)
            break
    env.unwrapped.agent_dir = random.randint(0, 3)
    obs = env.unwrapped.gen_obs()
    state = get_state(obs)

    total_reward = 0

    for step in range(max_steps):
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)

        action = choose_action(state)
        next_obs, reward, done, truncated, _ = env.step(action)
        next_state = get_state(next_obs)

        if next_state not in q_table:
            q_table[next_state] = np.zeros(n_actions)

        # Q-learning update
        q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                  alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)


env.close()

# Rendering after Training
env = gym.make(minigrid, render_mode="human")

for episode in range(20):
    obs, _ = env.reset()

    while True:
        x = random.randint(0, env.unwrapped.width - 1)
        y = random.randint(0, env.unwrapped.height - 1)
        if env.unwrapped.grid.get(x, y) is None:
            env.unwrapped.agent_pos = (x, y)
            break

    env.unwrapped.agent_dir = random.randint(0, 3)
    obs = env.unwrapped.gen_obs()
    state = get_state(obs)

    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table.get(state, np.zeros(n_actions)))
        next_obs, reward, done, truncated, _ = env.step(action)
        state = get_state(next_obs)
        if done or truncated:
            print(f"Episode {episode} finished with reward: {reward}")
            break

env.close()

