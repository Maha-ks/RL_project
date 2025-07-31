import gymnasium as gym
import minigrid

def create_empty_env(render_mode):
    return gym.make("MiniGrid-Empty-5x5-v0", render_mode=render_mode)
