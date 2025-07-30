import gymnasium as gym
import minigrid
import numpy as np
import random
from tqdm import trange


alpha = 0.1
gamma = 0.99
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.999
num_episodes = 5000
max_steps = 200

env_name = "MiniGrid-DoorKey-5x5-v0"
env = gym.make(env_name, render_mode=None)
n_actions = env.action_space.n # 7 possible actions in MiniGrid

# Q-Table
q_table = {}

def get_state(obs):
    """Better state: includes visual + agent position + direction + has_key"""
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir  
    has_key = int(env.unwrapped.carrying is not None)
    image = tuple(obs["image"].flatten())
    return image + agent_pos + (agent_dir, has_key)

#Epsilon-greedy action selection
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    return np.argmax(q_table.get(state, np.zeros(n_actions)))


rewards = []
# Training loop
for episode in trange(num_episodes):
    obs, _ = env.reset()
    state = get_state(obs)
    total_reward = 0
    door_opened_once = False

    for step in range(max_steps):
        if state not in q_table:
            q_table[state] = np.zeros(n_actions)

        action = choose_action(state)
        next_obs, reward, done, truncated, _ = env.step(action)

        # Reward shaping : crucial in sparse-reward environments like DoorKey
        # +0.3 for picking up the key
        # +1.0 for first successful door open
        # -0.1 for re-toggling after door is open
        # +0.6 for walking past the door
        # + (1/distance_to_goal) * 0.5 to encourage goal-seeking
        # -0.05 penalty for spinning in place (left/right)
        
        
        has_key = env.unwrapped.carrying is not None
        grid = env.unwrapped.grid
        width, height = grid.width, grid.height

        door_open = False
        door_pos = None
        goal_pos = None

        for x in range(width):
            for y in range(height):
                cell = grid.get(x, y)
                if cell and cell.type == 'door':
                    if cell.is_open:
                        door_open = True
                    door_pos = (x, y)
                elif cell and cell.type == 'goal':
                    goal_pos = (x, y)

        front_cell = env.unwrapped.front_pos
        cell_in_front = grid.get(*front_cell)
        agent_pos = tuple(env.unwrapped.agent_pos)

        # Picked up key
        if has_key:
            reward += 0.3

        # First door open
        if action == 5 and has_key and cell_in_front and cell_in_front.type == 'door':
            if cell_in_front.is_open and not door_opened_once:
                reward += 1.0
                #print(f"Door opened at episode {episode}, step {step}")
                door_opened_once = True
            elif door_opened_once:
                reward -= 0.1  

        # Entering the room
        if door_open and door_pos and agent_pos != door_pos:
            reward += 0.6

        # Reward for approaching goal
        if door_open and goal_pos:
            dist = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
            shaped = (1.0 / (dist + 1e-5)) * 0.5  # boosted
            reward += shaped

        # Penalty for turning/spinning 
        if action in [0, 1]:  # turn left/right
            reward -= 0.05

        # Q-update
        next_state = get_state(next_obs)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(n_actions)

        q_table[state][action] = (1 - alpha) * q_table[state][action] + \
            alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)

env.close()

env = gym.make(env_name, render_mode="human")
for episode in range(10):
    obs, _ = env.reset()
    state = get_state(obs)
    for step in range(50):
        env.render()
        action = np.argmax(q_table.get(state, np.zeros(n_actions)))
        next_obs, reward, done, truncated, _ = env.step(action)
        state = get_state(next_obs)
        if done or truncated:
            print(f"Eval episode {episode} finished with reward: {reward}")
            break
env.close()
