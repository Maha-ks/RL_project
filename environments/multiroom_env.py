import gymnasium as gym
import minigrid

def create_multiroom_env(render_mode):
    return gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode=render_mode)

