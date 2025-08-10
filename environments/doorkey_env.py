import gymnasium as gym
import minigrid

def create_doorkey_env(render_mode):
    return gym.make("MiniGrid-DoorKey-6x6-v0", render_mode=render_mode)
