import gymnasium as gym
import minigrid

def create_unlock_env(render_mode):
    return gym.make("MiniGrid-Unlock-v0", render_mode=render_mode)
