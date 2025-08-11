import os
import yaml
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from environments import get_environment
from utils.state_utils import get_state
from utils.save_utils import load_q_table


def evaluate_policy(
    env, 
    q_table: Dict, 
    n_episodes: int = 30, 
    max_steps: int = 20
) -> Dict[str, np.ndarray]:
    """
    Evaluate a loaded Q-table greedily (argmax) for n_episodes.
    Returns per-episode arrays for reward, steps, and success flag.
    """
    rewards: List[float] = []
    steps_list: List[int] = []
    successes: List[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)

        ep_reward = 0.0
        ep_steps = 0
        done = False
        truncated = False

        for _ in range(max_steps):
            # Greedy action from Q-table
            q_vals = q_table.get(state, np.zeros(env.action_space.n, dtype=float))
            action = int(np.argmax(q_vals))

            next_obs, r, done, truncated, _info = env.step(action)
            ep_reward += float(r)
            ep_steps += 1

            state = get_state(env, next_obs)
            if done or truncated:
                break

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        successes.append(1 if done else 0)

    return {
        "rewards": np.array(rewards, dtype=float),
        "steps": np.array(steps_list, dtype=int),
        "successes": np.array(successes, dtype=int),
    }


def summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary stats for a set of episodes."""
    rewards = metrics["rewards"]
    steps = metrics["steps"]
    successes = metrics["successes"]

    success_mask = successes.astype(bool)
    success_rate = successes.mean() if len(successes) else 0.0
    avg_steps = int(round(steps.mean())) if len(steps) else 0
    avg_reward = rewards.mean() if len(rewards) else 0.0
    avg_steps_success = (
        int(round(steps[success_mask].mean())) if success_mask.any() else None
    )

    return {
        "avg_reward": float(avg_reward),
        "avg_steps": avg_steps,
        "success_rate": float(success_rate),
        "avg_steps_success": avg_steps_success,
    }

def main():
    # Load experiment config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    env_name = config["env_name"]
    strategy = config["strategy"]
    eval_episodes = 30
    max_steps = config.get("training", {}).get("max_steps", 20)

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

    # Where to save evaluation outputs
    outdir = os.path.join("results", "eval", env_name, strategy.replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)

    # Accumulators across seeds
    per_seed_rows = []

    base_env = get_environment(env_name, render_mode=None)

    for seed in seeds:
        # loading the Q-table for this seed
        try:
            q_table = load_q_table(env_name, strategy, seed)
        except Exception as e:
            print(f"[WARN] Could not load Q-table for seed {seed}: {e}")
            continue

        base_env.reset(seed=seed)

        # Evaluate
        ep_metrics = evaluate_policy(
            env=base_env,
            q_table=q_table,
            n_episodes=eval_episodes,
            max_steps=max_steps,
        )

        # Summarize this seed
        summary = summarize_metrics(ep_metrics)
        row = {
            "seed": seed,
            "episodes": eval_episodes,
            "avg_reward": round(summary["avg_reward"], 3),
            "avg_steps": summary["avg_steps"], 
            "success_rate": round(summary["success_rate"], 3),
            "avg_steps_success": summary["avg_steps_success"],
        }

        per_seed_rows.append(row)

        print(
            f"Seed {seed} — "
            f"AvgReward: {row['avg_reward']}, "
            f"AvgSteps: {row['avg_steps']}, "
            f"SuccessRate: {row['success_rate']}, "
            f"AvgSteps@Success: {row['avg_steps_success']}"
        )

    base_env.close()

    if not per_seed_rows:
        print("No seeds evaluated. Check that Q-tables exist and load_q_table is configured correctly.")
        return

    # Aggregate across seeds for the strategy
    avg_reward_vals = np.array([r["avg_reward"] for r in per_seed_rows], dtype=float)
    avg_steps_vals = np.array([r["avg_steps"] for r in per_seed_rows], dtype=float)
    success_rate_vals = np.array([r["success_rate"] for r in per_seed_rows], dtype=float)

    #
    steps_succ_vals = np.array(
        [r["avg_steps_success"] for r in per_seed_rows if r["avg_steps_success"] is not None],
        dtype=float,
    )

    overall = {
        "seed": "overall",
        "episodes": eval_episodes * len(per_seed_rows),
        "avg_reward": round(float(avg_reward_vals.mean()), 3),
        "avg_steps": int(round(avg_steps_vals.mean())),
        "success_rate": round(float(success_rate_vals.mean()), 3),
        "avg_steps_success": int(round(steps_succ_vals.mean())) if steps_succ_vals.size else None,
    }


    print("\n=== Strategy-level summary across seeds ===")
    print(
        f"{env_name} • {strategy} • {len(per_seed_rows)} seeds\n"
        f"AvgReward: {overall['avg_reward']} | "
        f"AvgSteps: {overall['avg_steps']} | "
        f"SuccessRate: {overall['success_rate']} | "
        f"AvgSteps@Success: {overall['avg_steps_success']}"
    )

    #Save CSV
    import csv
    csv_path = os.path.join(outdir, "evaluation_summary.csv")
    fieldnames = ["seed", "episodes", "avg_reward", "avg_steps", "success_rate", "avg_steps_success"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow(row)
        writer.writerow(overall)

    print(f"\nSaved summary to: {csv_path}")
def plot_all_strategies(strategies=None, seeds=None):
    """
    Append-only: compares multiple strategies using the already-saved Q-tables.
    Saves two plots under results/plots/<env_name>/.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env_name = cfg["env_name"]
    eval_episodes = 30                      
    max_steps = cfg.get("training", {}).get("max_steps", 20)

    if strategies is None:
        strategies = ["decay", "entropy", "count", "novelty"]
    if seeds is None:
        seeds = list(range(10))               

    outdir = os.path.join("results", "plots", env_name)
    os.makedirs(outdir, exist_ok=True)

    env = get_environment(env_name, render_mode=None)

    r_means, r_stds = [], []
    s_means, s_stds = [], []

    for strat in strategies:
        rs, ss = [], []
        for seed in seeds:
            try:
                qtab = load_q_table(env_name, strat, seed) #Load saved q tables
            except Exception as e:
                print(f"[WARN] Missing Q-table for {strat=} {seed=}: {e}")
                continue
            env.reset(seed=seed)
            metrics = evaluate_policy(env, qtab, n_episodes=eval_episodes, max_steps=max_steps)
            summ = summarize_metrics(metrics)
            rs.append(summ["avg_reward"])
            ss.append(summ["success_rate"])

        if rs:
            r_means.append(float(np.mean(rs))); r_stds.append(float(np.std(rs)))
            s_means.append(float(np.mean(ss))); s_stds.append(float(np.std(ss)))
        else:
            r_means.append(np.nan); r_stds.append(np.nan)
            s_means.append(np.nan); s_stds.append(np.nan)

    env.close()

    # Plot 1: Average Reward 
    x = np.arange(len(strategies))
    plt.figure(figsize=(9,5))
    plt.errorbar(x, r_means, yerr=r_stds, fmt='o-', capsize=4)
    plt.xticks(x, strategies)
    plt.ylabel("Average Reward (eval)")
    plt.title(f"{env_name} — Average Reward across strategies (mean ± std)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    p1 = os.path.join(outdir, "avg_reward_by_strategy.png")
    plt.savefig(p1, dpi=300, bbox_inches="tight"); plt.close()

    # Plot 2: Success Rate 
    plt.figure(figsize=(9,5))
    plt.errorbar(x, s_means, yerr=s_stds, fmt='o-', capsize=4)
    plt.xticks(x, strategies)
    plt.ylabel("Success Rate (0–1)")
    plt.title(f"{env_name} — Success Rate across strategies (mean ± std)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    p2 = os.path.join(outdir, "success_rate_by_strategy.png")
    plt.savefig(p2, dpi=300, bbox_inches="tight"); plt.close()

    print(f"\nSaved comparison plots:\n- {p1}\n- {p2}")

if __name__ == "__main__":
    main()
    plot_all_strategies(
        strategies=["decay", "entropy", "count", "novelty"],
        seeds=list(range(10))
    )
