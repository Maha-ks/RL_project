import numpy as np

def get_state(env, obs):
    agent_pos = tuple(env.unwrapped.agent_pos)
    agent_dir = env.unwrapped.agent_dir
    has_key = int(env.unwrapped.carrying is not None)
    image = tuple(obs["image"].flatten())
    return image + agent_pos + (agent_dir, has_key)

def entropy(probs):
    probs = np.clip(probs, 1e-6, 1.0)
    return -np.sum(probs * np.log(probs))

