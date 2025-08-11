from .doorkey_env import create_doorkey_env
from .unlock_env import create_unlock_env

def get_environment(name, render_mode):
    print(name)
    if name == "DoorKey":
        return create_doorkey_env(render_mode)
    elif name == "Unlock":
        return create_unlock_env(render_mode)
    else:
        raise ValueError(f"Unknown environment: {name}")
