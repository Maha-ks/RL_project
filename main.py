import yaml
import numpy as np
import random
import os

from agent.q_learning_agent import train_q_learning
from utils.save_utils import save_q_table
from environments import get_environment
from utils.state_utils import get_state

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def evaluate_agent(env, q_table, get_state, n_episodes=10):
    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0
        for _ in range(100):
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            action = int(np.argmax(q_vals))

            next_obs, reward, done, truncated, _ = env.step(action)
            state = get_state(env, next_obs)
            total_reward += reward
            if done or truncated:
                break
        print(f"Eval Episode {ep+1}: Total reward = {total_reward:.2f}")
    env.close()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run for multiple seeds
seeds = [0, 1, 2, 3, 4] 

for seed in seeds:
    print(f"\n=== Running for seed {seed} ===")
    set_global_seed(seed)

    # Set up environment with seed
    env = get_environment(config["env_name"], None)
    env.reset(seed=seed)

    # Train
    q_table, rewards = train_q_learning(env, config["strategy"], config["training"], seed=seed)
    save_q_table(q_table, config["env_name"], config["strategy"], seed)

    # Evaluate
    eval_env = get_environment(config["env_name"], "human")
    eval_env.reset(seed=seed)
    evaluate_agent(eval_env, q_table, get_state)

#from utils.save_utils import load_q_table
# q_table = load_q_table("MiniGrid-DoorKey-5x5-v0", "novelty", seed=0)
