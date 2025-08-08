import yaml
import numpy as np
from agent.q_learning_agent import train_q_learning
from utils.save_utils import save_q_table
from environments import get_environment
from utils.state_utils import get_state


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env = get_environment(config["env_name"], None)
q_table, rewards = train_q_learning(env, config["strategy"], config["training"])

env = get_environment(config["env_name"], "human")

# Evaluate agent
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

# Run evaluation
evaluate_agent(env, q_table, get_state)
