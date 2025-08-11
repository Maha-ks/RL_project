import yaml, csv, random, os
import numpy as np
import matplotlib.pyplot as plt

from agent.q_learning_agent import train_q_learning
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
    for k in ["novelty_weights"]:
        if k in cfg:
            print(f"{k}: {cfg[k]}")
    print("="*48)

def rolling_success(successes, window=10):
    if not successes: return []
    out = []
    for i in range(len(successes)):
        lo = max(0, i - window)
        out.append(np.mean(successes[lo:i+1]))
    return out

def average_series(list_of_lists):
    """
    Averages a list of 1D sequences element-wise.
    Truncates all to the minimum length so shapes match.
    Returns a Python list.
    """
    if not list_of_lists:
        return []
    min_len = min(len(x) for x in list_of_lists if x is not None)
    if min_len == 0:
        return []
    arr = np.stack([np.asarray(x[:min_len], dtype=float) for x in list_of_lists])
    return arr.mean(axis=0).tolist()


def main():
    # Load baseline (fixed training params) and tuning spec (only weight sets + general info)
    with open("config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)   # baseline training lives here
    with open("tuning_config.yaml", "r") as f:
        tune = yaml.safe_load(f)

    strategy  = tune["strategy"]
    env_name  = tune["env_name"]

    seeds = tune.get("seeds", [tune.get("seed", 0)])

    # Only tune novelty weights
    weight_sets = tune["novelty_weight_sets"]
    N = tune.get("num_random_configs", len(weight_sets))
    all_combos = list(weight_sets)
    if N > len(all_combos):
        N = len(all_combos)
    sampled = random.sample(all_combos, N)

    # Fixed baseline params
    base_train = base_cfg["training"]
    BASE_ALPHA     = base_train["alpha"]
    BASE_EPS       = base_train["epsilon"]
    BASE_EPS_DECAY = base_train["epsilon_decay"]
    BASE_GAMMA     = base_train["gamma"]

    pretty_print_cfg("Baseline Configuration (from config.yaml)", base_train)

    # Outputs
    os.makedirs("tune_logs", exist_ok=True)
    csv_path = os.path.join("tune_logs", f"{strategy}_weights_only_tuning_results.csv")
    tuning_root = "tuning"; os.makedirs(tuning_root, exist_ok=True)


    # Prepare CSV and write header once
    with open(csv_path, "w", newline="") as cf:
        wcsv = csv.writer(cf)
        wcsv.writerow([
            "config_index","env","strategy_label",
            "w_entropy","w_progress","w_variance",
            "alpha","epsilon","epsilon_decay","gamma",
            "final_reward","convergence_episode","avg_unique_states"
        ])

    # Run configs one by one, and for each config run all seeds 
    for idx, w_set in enumerate(sampled, start=1):
        print(f"\n=== Running ALL seeds for cfg {idx} | w={w_set} ===")

        strat_label = (
            f"{strategy}_wE{w_set['entropy']}_wP{w_set['progress']}_wV{w_set['variance']}"
            f"_a{BASE_ALPHA}_eps{BASE_EPS}_decay{BASE_EPS_DECAY}_g{BASE_GAMMA}"
        )

        final_rewards = []
        conv_eps      = []
        avg_uniques   = []
        rewards_list  = []
        unique_states_list = []
        successes_list = []

        for seed in seeds:
            print(f"\n--- [cfg {idx}] Seed {seed} ---")
            set_global_seed(seed)

            env = get_environment(env_name, None)
            env.reset(seed=seed)

            cfg = base_train.copy()
            cfg["novelty_weights"] = w_set

            _, metrics = train_q_learning(env, strategy, cfg, seed=seed)
            rewards       = metrics["rewards"]
            successes     = metrics["successes"]
            unique_states = metrics["unique_states"]


            final_reward   = mean_last_k(rewards, k=100)
            conv_ep        = convergence_episode_from_rewards(rewards, threshold=0.6, window=50)
            avg_unique_all = (np.mean(unique_states) if unique_states else 0.0)

            final_rewards.append(final_reward)
            avg_uniques.append(avg_unique_all)
            if conv_ep is not None:
                conv_eps.append(conv_ep)

            rewards_list.append(list(rewards))
            unique_states_list.append(list(unique_states))
            successes_list.append(list(successes))

        # Averaged plots
        cfg_dir = os.path.join(tuning_root, str(idx)); os.makedirs(cfg_dir, exist_ok=True)

        avg_rewards = average_series(rewards_list)
        avg_unique  = average_series(unique_states_list)
        avg_roll_succ = average_series([rolling_success(s, window=10) for s in successes_list])

        plot_series(avg_rewards, f"Total Reward per Episode (cfg {idx})", "Reward",
                    os.path.join(cfg_dir, "total_reward_per_episode.png"), smooth_window=50)
        plot_series(avg_unique, f"Unique States Visited per Episode (cfg {idx})", "# Unique States",
                    os.path.join(cfg_dir, "unique_states_per_episode.png"), smooth_window=50)
        if len(avg_roll_succ) > 0:
            plot_series(avg_roll_succ, f"Rolling Success Rate (w=10) (cfg {idx})", "Success Rate",
                        os.path.join(cfg_dir, "rolling_success_rate.png"))

        # Save config.yaml once per config
        with open(os.path.join(cfg_dir, "config.yaml"), "w") as yf:
            yaml.safe_dump({
                "env_name": env_name,
                "strategy": strategy,
                "seeds": seeds,
                "novelty_weights": w_set,
                "alpha": BASE_ALPHA,
                "epsilon": BASE_EPS,
                "epsilon_decay": BASE_EPS_DECAY,
                "gamma": BASE_GAMMA,
                "base_training": base_train
            }, yf)

        # Append CSV row
        mean_final  = float(np.mean(final_rewards)) if final_rewards else float("nan")
        mean_unique = float(np.mean(avg_uniques)) if avg_uniques else float("nan")
        mean_conv   = (float(np.mean(conv_eps)) if conv_eps else None)

        with open(csv_path, "a", newline="") as cf:
            wcsv = csv.writer(cf)
            wcsv.writerow([
                idx, env_name, strat_label,
                w_set["entropy"], w_set["progress"], w_set["variance"],
                BASE_ALPHA, BASE_EPS, BASE_EPS_DECAY, BASE_GAMMA,
                f"{mean_final:.6f}",
                (f"{mean_conv:.2f}" if mean_conv is not None else "NA"),
                f"{mean_unique:.6f}",
            ])

if __name__ == "__main__":
    main()
