import yaml
import numpy as np
import random
import os
import matplotlib.pyplot as plt
os.makedirs("plots/combined", exist_ok=True)

from agent.q_learning_agent import train_q_learning
from utils.save_utils import save_q_table
from environments import get_environment
from utils.state_utils import get_state

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def evaluate_agent(env, q_table, get_state, n_episodes=10):
    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0
        for _ in range(100):
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            action = int(np.argmax(q_vals))

            next_obs, reward, done, truncated, _ = env.step(action)
            state = get_state(env, next_obs)
            total_reward += reward
            if done or truncated:
                break
        print(f"Eval Episode {ep+1}: Total reward = {total_reward:.2f}")
    env.close()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run for multiple seeds
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
all_rewards = []
all_successes = []
all_unique_states = []

for seed in seeds:
    print(f"\n=== Running for seed {seed} ===")
    set_global_seed(seed)

    env = get_environment(config["env_name"], None)
    env.reset(seed=seed)

    q_table, metrics = train_q_learning(
        env, config["strategy"], config["training"], seed=seed
    )
    all_rewards.append(metrics["rewards"])
    all_successes.append(metrics["successes"])
    all_unique_states.append(metrics["unique_states"])

    save_q_table(q_table, config["env_name"], config["strategy"], seed)

    eval_env = get_environment(config["env_name"], None)
    eval_env.reset(seed=seed)
    evaluate_agent(eval_env, q_table, get_state)

all_rewards = np.array(all_rewards)          
all_successes = np.array(all_successes)    
all_unique_states = np.array(all_unique_states)

mean_rewards = all_rewards.mean(axis=0)
std_rewards = all_rewards.std(axis=0)

mean_success_rate = all_successes.mean(axis=0)      
std_success_rate = all_successes.std(axis=0)

mean_unique = all_unique_states.mean(axis=0)
std_unique = all_unique_states.std(axis=0)

episodes = np.arange(mean_rewards.shape[0])

# Optional smoothing for readability
def smooth(x, w=50):
    if len(x) < w:
        return x
    kern = np.ones(w) / w
    return np.convolve(x, kern, mode="same")

sm_mean_rewards = smooth(mean_rewards, w=50)
sm_success_rate = smooth(mean_success_rate, w=10)
sm_unique = smooth(mean_unique, w=10)

title_base = f"{config['env_name']} • {config['strategy']} • {len(seeds)} seeds"
outdir = os.path.join("plots", "combined", config["env_name"], config["strategy"].replace(" ", "_"))
os.makedirs(outdir, exist_ok=True)

last_n = min(100, mean_rewards.shape[0])

# Reward
final_reward_mean = mean_rewards[-last_n:].mean()
final_reward_std  = mean_rewards[-last_n:].std()

# Success rate
final_success_mean = mean_success_rate[-last_n:].mean()
final_success_std  = std_success_rate[-last_n:].mean()

# Unique states (int rounded)
final_unique_mean = int(round(mean_unique[-last_n:].mean()))
final_unique_std  = int(round(mean_unique[-last_n:].std()))

# 1) Reward (mean ± std)
plt.figure()
plt.plot(episodes, sm_mean_rewards, label="Mean reward (smoothed)")
plt.fill_between(episodes,
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.2, label="±1 std")
plt.legend(loc="lower left")
plt.annotate(
    f"Final mean: {final_reward_mean:.3f}\nFinal std: {final_reward_std:.3f}",
    xy=(0.99, 0.01), xycoords='axes fraction',
    ha='right', va='bottom', fontsize=9,
    bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray')
)
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"Reward over Episodes — {title_base}")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(outdir, "reward_mean_std.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2) Success rate across seeds (mean ± std)
plt.figure()
plt.plot(episodes, sm_success_rate, label="Mean success rate (smoothed)")
plt.fill_between(episodes,
                 mean_success_rate - std_success_rate,
                 mean_success_rate + std_success_rate,
                 alpha=0.2, label="±1 std")
plt.legend(loc="lower left")
plt.annotate(
    f"Final mean: {final_success_mean:.3f}\nFinal std: {final_success_std:.3f}",
    xy=(0.99, 0.01), xycoords='axes fraction',
    ha='right', va='bottom', fontsize=9,
    bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray')
)
plt.xlabel("Episode"); plt.ylabel("Success rate (0–1)"); plt.title(f"Success Rate — {title_base}")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(outdir, "success_rate_mean_std.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3) Unique states visited (mean ± std) — integers
plt.figure()
plt.plot(episodes, sm_unique, label="Mean unique states (smoothed)")
plt.fill_between(episodes,
                 mean_unique - std_unique,
                 mean_unique + std_unique,
                 alpha=0.2, label="±1 std")
plt.legend(loc="lower left")
plt.annotate(
    f"Final mean: {final_unique_mean}\nFinal std: {final_unique_std}",
    xy=(0.99, 0.99), xycoords='axes fraction',
    ha='right', va='top', fontsize=9,
    bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray')
)
plt.xlabel("Episode"); plt.ylabel("# Unique states"); plt.title(f"Exploration — {title_base}")
plt.grid(True); plt.legend()
plt.savefig(os.path.join(outdir, "unique_states_mean_std.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"[Summary over last {last_n} episodes]")
print(f"Reward:  mean={final_reward_mean:.3f} ± {final_reward_std:.3f}")
print(f"Success: mean={final_success_mean:.3f} ± {final_success_std:.3f}")
print(f"Unique:  mean={final_unique_mean} ± {final_unique_std}")
