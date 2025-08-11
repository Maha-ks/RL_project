import yaml, csv, itertools, random, os
import numpy as np
import matplotlib.pyplot as plt

from agent.q_learning_agent import train_q_learning
from utils.save_utils import save_q_table
from environments import get_environment

def set_global_seed(seed):
    np.random.seed(seed); random.seed(seed)

def convergence_episode_from_rewards(rewards, threshold=0.6, window=50):
    if len(rewards) < window:
        return None
    rr = np.convolve(rewards, np.ones(window)/window, mode="valid")
    for i, v in enumerate(rr):
        if v >= threshold:
            return i + window - 1
    return None

def mean_last_k(rewards, k=100):
    if not rewards:
        return -1e9
    k = min(k, len(rewards))
    return float(np.mean(rewards[-k:]))

def plot_series(y, title, ylabel, outpath, smooth_window=None):
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y, alpha=0.6, label="raw")
    if smooth_window and len(y) >= smooth_window:
        s = np.convolve(y, np.ones(smooth_window)/smooth_window, mode="valid")
        plt.plot(np.arange(smooth_window-1, len(y)), s, label=f"smoothed (w={smooth_window})")
    plt.title(title)
    plt.xlabel("Episode"); plt.ylabel(ylabel)
    plt.grid(True); plt.legend()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def pretty_print_cfg(title, cfg):
    print(f"\n=== {title} ===")
    for k in ["alpha","gamma","epsilon","min_epsilon","epsilon_decay","beta","num_episodes","max_steps","novelty_weights"]:
        if k in cfg:
            print(f"{k}: {cfg[k]}")
    print("="*48)

def main():
    # Load configs
    with open("config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open("tuning_config.yaml", "r") as f:
        tune = yaml.safe_load(f)
    with open("baseline_config.yaml", "r") as f:
        baseline_cfg = yaml.safe_load(f)

    strategy  = tune["strategy"]
    env_name  = tune["env_name"]
    seed      = 0

    weight_sets = tune["novelty_weight_sets"]
    alphas      = tune["alpha_values"]
    eps_starts  = tune["epsilon_starts"]
    eps_decays  = tune["epsilon_decays"]
    gammas      = tune["gamma_values"]
    N           = tune.get("num_random_configs", 20)

    # ---------- Print baseline ----------
    pretty_print_cfg("Baseline Configuration (from baseline_config.yaml)", baseline_cfg["training"])

    # ---------- Sample configs ----------
    all_combos = list(itertools.product(weight_sets, alphas, eps_starts, eps_decays, gammas))
    if N > len(all_combos): N = len(all_combos)
    sampled = random.sample(all_combos, N)

    #Outputs
    os.makedirs("tune_logs", exist_ok=True)
    csv_path = os.path.join("tune_logs", f"{strategy}_random_tuning_results.csv")
    tuning_root = "tuning"; os.makedirs(tuning_root, exist_ok=True)

    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([
            "config_index","env","strategy_label",
            "w_entropy","w_progress","w_variance",
            "alpha","epsilon","epsilon_decay","gamma","seed",
            "final_reward","convergence_episode","avg_unique_states"
        ])

    results_summary = []

    # Tuning loop
    for idx, (w_set, alpha, eps_start, eps_decay, gamma) in enumerate(sampled, start=1):
        print(f"\n=== [{idx}/{N}] w={w_set}, alpha={alpha}, eps={eps_start}, decay={eps_decay}, gamma={gamma}, seed={seed} ===")
        set_global_seed(seed)

        env = get_environment(env_name, None)
        env.reset(seed=seed)

        cfg = base_cfg["training"].copy()
        cfg["alpha"] = alpha
        cfg["epsilon"] = eps_start
        cfg["epsilon_decay"] = eps_decay
        cfg["gamma"] = gamma
        cfg["novelty_weights"] = w_set

        q_table, metrics = train_q_learning(env, strategy, cfg, seed=seed)
        rewards       = metrics["rewards"]
        successes     = metrics["successes"]
        unique_states = metrics["unique_states"]

        strat_label = (f"{strategy}_wE{w_set['entropy']}_wP{w_set['progress']}_wV{w_set['variance']}"
                       f"_a{alpha}_eps{eps_start}_decay{eps_decay}_g{gamma}")
        save_q_table(q_table, env_name, strat_label, seed)

        final_reward   = mean_last_k(rewards, k=100)
        conv_ep        = convergence_episode_from_rewards(rewards, threshold=0.6, window=50)
        avg_unique_all = int(round(np.mean(unique_states))) if unique_states else 0

        with open(csv_path, "a", newline="") as cf:
            w = csv.writer(cf)
            w.writerow([
                idx, env_name, strat_label,
                w_set["entropy"], w_set["progress"], w_set["variance"],
                alpha, eps_start, eps_decay, gamma, seed,
                f"{final_reward:.6f}", (conv_ep if conv_ep is not None else "NA"), avg_unique_all
            ])

        cfg_dir = os.path.join(tuning_root, str(idx)); os.makedirs(cfg_dir, exist_ok=True)
        plot_series(rewards,       f"Total Reward per Episode (cfg {idx})",          "Reward",
                    os.path.join(cfg_dir, "total_reward_per_episode.png"), smooth_window=50)
        plot_series(unique_states, f"Unique States Visited per Episode (cfg {idx})", "# Unique States",
                    os.path.join(cfg_dir, "unique_states_per_episode.png"), smooth_window=50)
        if len(successes) > 0:
            window = 10
            rs = [np.mean(successes[max(0, i - window):i + 1]) for i in range(len(successes))]
            plot_series(rs, f"Rolling Success Rate (w=10) (cfg {idx})", "Success Rate",
                        os.path.join(cfg_dir, "rolling_success_rate.png"))

        with open(os.path.join(cfg_dir, "config.yaml"), "w") as yf:
            yaml.safe_dump({
                "env_name": env_name,
                "strategy": strategy,
                "seed": seed,
                "novelty_weights": w_set,
                "alpha": alpha,
                "epsilon": eps_start,
                "epsilon_decay": eps_decay,
                "gamma": gamma,
                "base_training": base_cfg["training"]
            }, yf)

        score = final_reward - 0.0005 * (conv_ep if conv_ep is not None else 1e9)
        results_summary.append({
            "index": idx,
            "key": (w_set["entropy"], w_set["progress"], w_set["variance"], alpha, eps_start, eps_decay, gamma),
            "final_reward": final_reward,
            "conv_ep": conv_ep,
            "score": score
        })

    # print Top-2 
    results_summary.sort(key=lambda x: x["score"], reverse=True)
    top2 = results_summary[:2]
    print("\n================  TOP 2 CONFIGS  ================")
    for i, s in enumerate(top2, 1):
        wE, wP, wV, a, e0, ed, g = s["key"]
        why = [f"final reward={s['final_reward']:.3f}"]
        why.append(f"convergence ep={s['conv_ep']}" if s["conv_ep"] is not None else "no convergence")
        print(f"{i}. idx={s['index']} | wE={wE}, wP={wP}, wV={wV} | alpha={a} | eps0={e0} | decay={ed} | gamma={g}")
        print("   -> " + "; ".join(why))

    # Baseline vs Top-2 comparison
    def run_config_and_get_metrics(env_name, strategy, seed, base_training, ow):
        env = get_environment(env_name, None)
        env.reset(seed=seed)
        cfg = base_training.copy()
        cfg["alpha"] = ow.get("alpha", cfg["alpha"])
        cfg["epsilon"] = ow.get("epsilon", cfg["epsilon"])
        cfg["epsilon_decay"] = ow.get("epsilon_decay", cfg["epsilon_decay"])
        cfg["gamma"] = ow.get("gamma", cfg["gamma"])
        cfg["novelty_weights"] = ow.get("novelty_weights", cfg.get("novelty_weights"))
        _, metrics = train_q_learning(env, strategy, cfg, seed=seed)
        return metrics

    def rolling_success(successes, window=10):
        if not successes: return []
        out = []
        for i in range(len(successes)):
            lo = max(0, i - window)
            out.append(np.mean(successes[lo:i+1]))
        return out

    compare_dir = os.path.join("tuning", "compare_top2")
    os.makedirs(compare_dir, exist_ok=True)

    baseline_overrides = {
        "alpha": baseline_cfg["training"]["alpha"],
        "epsilon": baseline_cfg["training"]["epsilon"],
        "epsilon_decay": baseline_cfg["training"]["epsilon_decay"],
        "gamma": baseline_cfg["training"]["gamma"],
        "novelty_weights": baseline_cfg["training"]["novelty_weights"],
    }

    print("\n[Compare] Running baseline novelty config (from baseline_config.yaml)...")
    baseline_metrics = run_config_and_get_metrics(env_name, strategy, seed, baseline_cfg["training"], baseline_overrides)

    # Prepare two best configs
    best_cfgs = []
    for s in top2:
        wE, wP, wV, a, e0, ed, g = s["key"]
        best_cfgs.append({
            "label": f"cfg {s['index']}",
            "overrides": {
                "alpha": a, "epsilon": e0, "epsilon_decay": ed, "gamma": g,
                "novelty_weights": {"entropy": wE, "progress": wP, "variance": wV},
            }
        })

    curves = [{"label": "baseline", "m": baseline_metrics}]
    for item in best_cfgs:
        print(f"[Compare] Running {item['label']}...")
        m = run_config_and_get_metrics(env_name, strategy, seed, baseline_cfg["training"], item["overrides"])
        curves.append({"label": item["label"], "m": m})

    # Plot with baseline in black
    def plot_compare(lines, title, ylabel, outpath, smooth_window=None):
        plt.figure()
        for lab, y in lines:
            x = np.arange(len(y))
            if lab.lower() == "baseline":
                plt.plot(x, y, label=lab, color="black", linewidth=2.2, alpha=0.95)
                if smooth_window and len(y) >= smooth_window:
                    s = np.convolve(y, np.ones(smooth_window)/smooth_window, mode="valid")
                    plt.plot(np.arange(smooth_window-1, len(y)), s, color="black", linewidth=2.8, alpha=0.6)
            else:
                plt.plot(x, y, label=lab, alpha=0.9)
                if smooth_window and len(y) >= smooth_window:
                    s = np.convolve(y, np.ones(smooth_window)/smooth_window, mode="valid")
                    plt.plot(np.arange(smooth_window-1, len(y)), s, alpha=0.6)
        plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel)
        plt.grid(True); plt.legend()
        plt.savefig(outpath, dpi=300, bbox_inches="tight"); plt.close()

    reward_lines = [(c["label"], c["m"]["rewards"]) for c in curves]
    uniq_lines   = [(c["label"], c["m"]["unique_states"]) for c in curves]
    succ_lines   = [(c["label"], rolling_success(c["m"]["successes"], window=10)) for c in curves]

    plot_compare(reward_lines, "Average total reward (baseline vs. top-2)", "Reward",
                 os.path.join(compare_dir, "reward_compare.png"), smooth_window=50)
    plot_compare(uniq_lines, "Unique states per episode (baseline vs. top-2)", "# Unique states",
                 os.path.join(compare_dir, "unique_states_compare.png"), smooth_window=50)
    plot_compare(succ_lines, "Rolling success rate (w=10) (baseline vs. top-2)", "Success rate",
                 os.path.join(compare_dir, "success_rate_compare.png"))

    print(f"\n[Compare] Saved comparison plots to: {compare_dir}")

if __name__ == "__main__":
    main()
