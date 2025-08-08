import numpy as np

# === Count-based intrinsic bonus ===
def intrinsic_count_bonus(strategy, next_state, state_visits, beta):
    if strategy == "count":
        visit_count = state_visits[next_state]

        #print('Visit count: ',visit_count)
        #print('Count reward : ',beta / (np.sqrt(visit_count + 1)))
        return beta / (np.sqrt(visit_count + 1))
    return 0.0

def get_prev_pos(env, curr_pos, action):
    """
    Estimate the agent's previous position based on the action taken.
    Action space:
    0 = left turn
    1 = right turn
    2 = forward
    3 = pickup
    4 = drop
    5 = toggle
    6 = done
    """
    x, y = curr_pos

    # MiniGrid directions: 0=right, 1=down, 2=left, 3=up
    # We'll simulate "stepping backward" if the agent moved forward
    if action == 2:  # forward
        dir = env.agent_dir
        if dir == 0:  # right
            return (x - 1, y)
        elif dir == 1:  # down
            return (x, y - 1)
        elif dir == 2:  # left
            return (x + 1, y)
        elif dir == 3:  # up
            return (x, y + 1)

    # For other actions, assume the agent stayed in place
    return curr_pos

def apply_key_distance_shaping(env, shaped_reward, action):
    """
    Computes reward for getting closer to the key *based on predicted previous position*.
    """
    if env.carrying and env.carrying.type == 'key':
        return shaped_reward

    agent_pos = env.agent_pos
    prev_pos = get_prev_pos(env, agent_pos, action)

    # Locate key
    key_pos = None
    for i in range(env.width):
        for j in range(env.height):
            tile = env.grid.get(i, j)
            if tile and tile.type == 'key':
                key_pos = (i, j)
                break

    if key_pos:
        prev_dist = manhattan_dist(prev_pos, key_pos)
        curr_dist = manhattan_dist(agent_pos, key_pos)
        if curr_dist < prev_dist:
            shaped_reward += 0.03  # reward for progress
        else:
            shaped_reward -= 0.03

    return shaped_reward

def apply_goal_distance_shaping(env, shaped_reward, action):
    """
    Computes reward for getting closer to the goal *based on predicted previous position*.
    """
    if not (env.carrying and env.carrying.type == 'key'):
        return shaped_reward
        
    agent_pos = env.agent_pos
    prev_pos = get_prev_pos(env, agent_pos, action)

    # Locate goal
    goal_pos = None
    for i in range(env.width):
        for j in range(env.height):
            tile = env.grid.get(i, j)
            if tile and tile.type == 'goal':
                goal_pos = (i, j)
                break

    if goal_pos:
        prev_dist = manhattan_dist(prev_pos, goal_pos)
        curr_dist = manhattan_dist(agent_pos, goal_pos)
        if curr_dist < prev_dist:
            shaped_reward += 0.05
        else:
            shaped_reward -= 0.05

    return shaped_reward

def apply_goal_distance_shaping_toward_box(env, shaped_reward, action):
    """
    Computes reward for getting closer to the box (goal) only after the door is open.
    Uses predicted previous position based on the action taken.
    """
    agent_pos = env.agent_pos
    prev_pos = get_prev_pos(env, agent_pos, action)

    # Check if the door is open
    door_open = any(
        tile and tile.type == 'door' and tile.is_open
        for i in range(env.width)
        for j in range(env.height)
        if (tile := env.grid.get(i, j))
    )

    if not door_open:
        return shaped_reward

    # Locate the box
    box_pos = None
    for i in range(env.width):
        for j in range(env.height):
            tile = env.grid.get(i, j)
            if tile and tile.type == 'box':
                box_pos = (i, j)
                break
        if box_pos:
            break

    if box_pos:
        prev_dist = manhattan_dist(prev_pos, box_pos)
        curr_dist = manhattan_dist(agent_pos, box_pos)
        if curr_dist < prev_dist:
            shaped_reward += 0.05
        else:
            shaped_reward -= 0.05

    return shaped_reward


def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def shape_reward(
    env, action, reward, strategy, state_visits, next_state, beta, has_received_key_reward, prev_agent_pos
):
    env = env.unwrapped
    shaping_log = []

    # Step penalty
    reward -= 0.01
    shaping_log.append("- Step penalty: -0.01")

    if env.agent_pos == prev_agent_pos:
        reward -= 0.05
        shaping_log.append("- Idle penalty: -0.05")

    has_key = env.carrying and env.carrying.type == 'key'
    if has_key and not has_received_key_reward:
        reward += 0.3
        has_received_key_reward = True
        shaping_log.append("+ Picked up key: +0.3")

    door_unlocked = any(
        tile and tile.type == 'door' and tile.is_open
        for i in range(env.width)
        for j in range(env.height)
        if (tile := env.grid.get(i, j))
    )
    if door_unlocked:
        reward += 0.4
        shaping_log.append("+ Door unlocked: +0.4")

    cell = env.grid.get(*env.agent_pos)
    if cell and cell.type == 'goal':
        reward += 1.0
        shaping_log.append("+ Reached goal: +1.0")

    if env.carrying and env.carrying.type == 'box':
        shaped = 1 - 0.9 * (env.step_count / env.max_steps)
        reward += shaped
        shaping_log.append(f"+ Reached box: +{shaped:.3f}")

    bonus = intrinsic_count_bonus(strategy, next_state, state_visits, beta)
    if bonus > 0:
        reward += bonus
        shaping_log.append(f"+ Count-based bonus: +{bonus:.3f}")

    prev = reward
    reward = apply_key_distance_shaping(env, reward, action)
    if reward > prev:
        shaping_log.append(f"+ Closer to key: +{reward - prev:.3f}")
    elif reward < prev:
        shaping_log.append(f"- Farther from key: -{prev - reward:.3f}")

    prev = reward
    reward = apply_goal_distance_shaping(env, reward, action)
    if reward > prev:
        shaping_log.append(f"+ Closer to goal: +{reward - prev:.3f}")
    elif reward < prev:
        shaping_log.append(f"- Farther from goal: -{prev - reward:.3f}")

    prev = reward
    reward = apply_goal_distance_shaping_toward_box(env, reward, action)
    if reward > prev:
        shaping_log.append(f"+ Closer to box: +{reward - prev:.3f}")
    elif reward < prev:
        shaping_log.append(f"- Farther from box: -{prev - reward:.3f}")

    return reward, has_received_key_reward
