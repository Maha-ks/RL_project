import numpy as np

def save_q_table(q_table, path):
    np.save(path, q_table)

def load_q_table(path):
    return np.load(path, allow_pickle=True).item()
