def reset_env(env):
    """
    Resets the environment and returns the initial observation.
    """
    obs, _ = env.reset()
    return obs

def step_env(env, action):
    """
    Executes an action in the environment and returns the results.
    """
    return env.step(action)