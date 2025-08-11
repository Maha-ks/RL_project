import numpy as np
from collections import deque

LOG_FILE = "reward_log.txt"

def intrinsic_count_bonus(next_state, state_visits, beta):
    visit_count = state_visits[next_state]
    return beta / (np.sqrt(visit_count + 1))


def shortest_distance(env, start, target):
    """Shortest steps from start to target in MiniGrid, allowing target even if blocked."""
    sx, sy = start
    tx, ty = target
    W, H = env.width, env.height
    
    def passable(x, y, target):
        if not (0 <= x < W and 0 <= y < H):
            return False
        if (x, y) == target:
            return True  # always allow the target
        tile = env.grid.get(x, y)
        return not (tile and tile.type == "wall")

    if (sx, sy) == (tx, ty):
        return 0

    q = deque([((sx, sy), 0)])
    visited = {(sx, sy)}

    while q:
        (x, y), dist = q.popleft()
        
        for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            
            if (nx, ny) == (tx, ty):
                return dist + 1
            
            if (nx, ny) not in visited and passable(nx, ny, target):
                visited.add((nx, ny))
                q.append(((nx, ny), dist + 1))

    return None

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

def apply_key_distance_shaping(env, shaped_reward, action, door_unlocked, shaping_log):
    """
    Computes reward for getting closer to the key *based on predicted previous position*.
    """

    if door_unlocked:
        return shaped_reward, shaping_log
    
    if env.carrying and env.carrying.type == 'key':
        return shaped_reward, shaping_log

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
        if "DoorKey" in env.spec.id:
            prev_dist = shortest_distance(env, prev_pos, key_pos)
            curr_dist = shortest_distance(env, agent_pos, key_pos)

            shaping_log.append(f"~ Old key distance: {prev_dist:+.2f}")
            shaping_log.append(f"~ New key distance: {curr_dist:+.2f}")
                
        else:
            prev_dist = manhattan_dist(prev_pos, key_pos)
            curr_dist = manhattan_dist(agent_pos, key_pos)

        if curr_dist < prev_dist:
            shaped_reward += 0.03
        else:
            shaped_reward -= 0.03
        

    return shaped_reward, shaping_log

def apply_door_distance_shaping(env, shaped_reward, action, door_unlocked, shaping_log):
    """
    Reward for getting closer to the door (closed, matching key color),
    based on the previous position.
    """

    if door_unlocked:
        return shaped_reward, shaping_log

    # Only shape if the agent is holding a key
    if not (env.carrying and getattr(env.carrying, "type", None) == "key"):
        return shaped_reward, shaping_log

    agent_pos = tuple(env.agent_pos)
    prev_pos = get_prev_pos(env, agent_pos, action)

    # Locate goal
    door_position = None
    for i in range(env.width):
        for j in range(env.height):
            tile = env.grid.get(i, j)
            if tile and tile.type == 'door':
                door_position = (i, j)
                break

    # If we found no relevant doors, don't change the reward
    if door_position:
        if "DoorKey" in env.spec.id:
            prev_dist = shortest_distance(env, prev_pos, door_position)
            curr_dist = shortest_distance(env, agent_pos, door_position)
            shaping_log.append(f"~ Old door distance: {prev_dist:+.2f}")
            shaping_log.append(f"~ New door distance: {curr_dist:+.2f}")
                
        else:
            prev_dist = manhattan_dist(prev_pos, door_position)
            curr_dist = manhattan_dist(agent_pos, door_position)

        if curr_dist < prev_dist:
            shaped_reward += 0.05
        elif curr_dist > prev_dist:
            shaped_reward -= 0.05

    return shaped_reward, shaping_log


def apply_goal_distance_shaping(env, shaped_reward, action, door_unlocked, shaping_log):
    """
    Computes reward for getting closer to the goal *based on predicted previous position*.
    """

    if not door_unlocked:
        return shaped_reward, shaping_log
    
    if not (env.carrying and env.carrying.type == 'key'):
        return shaped_reward, shaping_log
        
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
        if "DoorKey" in env.spec.id:
            prev_dist = shortest_distance(env, prev_pos, goal_pos)
            curr_dist = shortest_distance(env, agent_pos, goal_pos)
            shaping_log.append(f"~ Old goal distance: {prev_dist:+.2f}")
            shaping_log.append(f"~ New goal distance: {curr_dist:+.2f}")
            if curr_dist < prev_dist:
                shaped_reward += 0.05
            else:
                shaped_reward -= 0.05

        else:
            prev_dist = manhattan_dist(prev_pos, goal_pos)
            curr_dist = manhattan_dist(agent_pos, goal_pos)
            if curr_dist < prev_dist:
                shaped_reward += 0.05
            else:
                shaped_reward -= 0.05

    return shaped_reward, shaping_log

def manhattan_dist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def shape_reward(
    env, action, reward, strategy, state_visits, next_state,
    beta, has_received_key_reward, prev_agent_pos, has_unlocked_door_reward, step
):
    env = env.unwrapped
    shaping_log = []

    # Step penalty
    reward -= 0.01
    shaping_log.append("- Step penalty: -0.01")

    shaping_log.append(f"- Old Pos: {prev_agent_pos}")
    shaping_log.append(f"- New Pos: {env.agent_pos}")
    # Penalty for not moving
    if env.agent_pos == prev_agent_pos:
        shaping_log.append("- Stayed in place: -0.05")
        reward -= 0.05
        return reward , has_received_key_reward, has_unlocked_door_reward

    has_key = env.carrying and getattr(env.carrying, "type", None) == "key"

    # Key pickup bonus
    if has_key and not has_received_key_reward:
        reward += 0.3
        has_received_key_reward = True
        shaping_log.append("+ Picked up key: +0.3")

    # Door unlocked bonus
    door_unlocked = any(
        tile and tile.type == 'door' and tile.is_open
        for i in range(env.width)
        for j in range(env.height)
        if (tile := env.grid.get(i, j))
    )
    if door_unlocked and not has_unlocked_door_reward:
        has_unlocked_door_reward = True
        reward += 0.4
        shaping_log.append("+ Door unlocked: +0.4")

    # Goal reached bonus
    cell = env.grid.get(*env.agent_pos)
    if cell and cell.type == 'goal':
        reward += 1.0
        shaping_log.append("+ Reached goal: +1.0")

    # Intrinsic bonus
    if strategy == "count":
        before_intrinsic = reward
        reward += intrinsic_count_bonus(next_state, state_visits, beta)
        if reward != before_intrinsic:
            shaping_log.append(f"~ Intrinsic bonus: {reward - before_intrinsic:+.2f}")

    # Key distance shaping
    before_key = reward
    reward, shaping_log = apply_key_distance_shaping(env, reward, action, door_unlocked, shaping_log)
    if reward != before_key:
        shaping_log.append(f"~ Key distance shaping: {reward - before_key:+.2f}")

    # Door distance shaping
    before_door = reward
    reward, shaping_log = apply_door_distance_shaping(env, reward, action, door_unlocked, shaping_log)
    if reward != before_door:
        shaping_log.append(f"~ Door distance shaping: {reward - before_door:+.2f}")

    # Goal distance shaping
    if "Unlock" in env.spec.id:
        before_goal = reward
        reward, shaping_log = apply_goal_distance_shaping(env, reward, action, door_unlocked, shaping_log)
        shaping_log.append(f"~ Goal distance shaping: {reward - before_goal:+.2f}")

    return reward, has_received_key_reward, has_unlocked_door_reward