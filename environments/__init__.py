from .doorkey_env import create_doorkey_env
from .empty_env import create_empty_env
from .multiroom_env import create_multiroom_env
from .unlock_env import create_unlock_env

def get_environment(name, render_mode):
    if name == "DoorKey":
        return create_doorkey_env(render_mode)
    elif name == "Empty":
        return create_empty_env(render_mode)
    elif name == "MultiRoom":
        return create_multiroom_env(render_mode)
    elif name == "Unlock":
        return create_unlock_env(render_mode)
    else:
        raise ValueError(f"Unknown environment: {name}")
