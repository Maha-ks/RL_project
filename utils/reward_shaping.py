import numpy as np

def shape_reward(env, action, reward, strategy, state_visits, next_state, beta):
    # Key pickup bonus
    has_key = env.unwrapped.carrying is not None
    if has_key:
        reward += 0.3

    # Door open bonus
    grid = env.unwrapped.grid
    front_cell = env.unwrapped.front_pos
    cell_in_front = grid.get(*front_cell)
    if action == 5 and has_key and cell_in_front and cell_in_front.type == 'door':
        if cell_in_front.is_open:
            reward += 1.0

    # Distance to goal bonus
    goal_pos = None
    agent_pos = tuple(env.unwrapped.agent_pos)
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and cell.type == 'goal':
                goal_pos = (x, y)
                break
    if goal_pos:
        dist = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))
        reward += (1.0 / (dist + 1e-5)) * 0.5

    # Penalty for turning
    if action in [0, 1]:
        reward -= 0.05

    # Count-based intrinsic reward
    if strategy == "count":
        visit_count = state_visits[next_state]
        reward += beta / (np.sqrt(visit_count + 1e-6))

    return reward
