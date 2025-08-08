import os
import pickle

def save_q_table(q_table, env_name, strategy, seed):
    strategy = strategy.replace(" ", "_").lower()
    env_name = env_name.lower()
    save_dir = os.path.join("q_tables", env_name, strategy, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "q_table.pkl")
    with open(path, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table(env_name, strategy, seed):
    strategy = strategy.replace(" ", "_").lower()
    env_name = env_name.lower()
    path = os.path.join("q_tables", env_name, strategy, f"seed_{seed}", "q_table.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No Q-table found at: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)
