import yaml, csv, itertools, random, os
import numpy as np
from agent.q_learning_agent import train_q_learning
from utils.save_utils import save_q_table
from environments import get_environment
from utils.state_utils import get_state

def set_global_seed(seed):
    np.random.seed(seed); random.seed(seed)

def evaluate_agent(env, q_table, get_state, n_episodes=10, max_steps=100):
    totals = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total = 0.0
        for _ in range(max_steps):
            q_vals = q_table.get(state, np.zeros(env.action_space.n))
            action = int(np.argmax(q_vals))
            next_obs, r, done, truncated, _ = env.step(action)
            state = get_state(env, next_obs)
            total += r
            if done or truncated:
                break
        totals.append(total)
    env.close()
    return float(np.mean(totals))

def convergence_episode_from_rewards(rewards, threshold=0.6, window=50): # threshold from the plot
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

def main():
    # Load configs
    with open("config.yaml", "r") as f:
        base_cfg = yaml.safe_load(f)
    with open("tuning_config.yaml", "r") as f:
        tune = yaml.safe_load(f)

    strategy = tune["strategy"]
    env_name = tune["env_name"]
    seed = 0  # fixed single seed

    weight_sets  = tune["novelty_weight_sets"]
    alphas       = tune["alpha_values"]
    eps_starts   = tune["epsilon_starts"]
    eps_decays   = tune["epsilon_decays"]
    gammas       = tune["gamma_values"]
    N            = tune.get("num_random_configs", 20)

    # Generate random combinations
    all_combos = list(itertools.product(weight_sets, alphas, eps_starts, eps_decays, gammas))
    if N > len(all_combos):
        N = len(all_combos)
    sampled = random.sample(all_combos, N)

    os.makedirs("tune_logs", exist_ok=True)
    csv_path = os.path.join("tune_logs", f"{strategy}_random_tuning_results.csv")

    with open(csv_path, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([
            "env","strategy_label","w_entropy","w_progress","w_variance",
            "alpha","epsilon","epsilon_decay","gamma","seed",
            "final_reward","convergence_episode","avg_unique_states"  
        ])

    results_summary = []

    for (w_set, alpha, eps_start, eps_decay, gamma) in sampled:
        print(f"\n=== w={w_set}, alpha={alpha}, eps={eps_start}, decay={eps_decay}, gamma={gamma}, seed={seed} ===")
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
        rewards = metrics["rewards"]
        u_states  = metrics["unique_states"]

        strat_label = (f"{strategy}_wE{w_set['entropy']}_wP{w_set['progress']}_wV{w_set['variance']}"
                       f"_a{alpha}_eps{eps_start}_decay{eps_decay}_g{gamma}")
        save_q_table(q_table, env_name, strat_label, seed)

        final_reward = mean_last_k(rewards, k=100)
        conv_ep = convergence_episode_from_rewards(rewards, threshold=0.6, window=50)
        avg_unique_all = int(round(np.mean(u_states))) if u_states else 0.0

        with open(csv_path, "a", newline="") as cf:
            w = csv.writer(cf)
            w.writerow([
                env_name, strat_label, w_set["entropy"], w_set["progress"], w_set["variance"],
                alpha, eps_start, eps_decay, gamma, seed,
                f"{final_reward:.6f}", (conv_ep if conv_ep is not None else "NA"),avg_unique_all
            ])

        score = final_reward - 0.0005 * (conv_ep if conv_ep is not None else 1e9)
        results_summary.append({
            "key": (w_set["entropy"], w_set["progress"], w_set["variance"], alpha, eps_start, eps_decay, gamma),
            "final_reward": final_reward,
            "conv_ep": conv_ep,
            "score": score
        })

    # Sort and show top 5
    results_summary.sort(key=lambda x: x["score"], reverse=True)
    top5 = results_summary[:5]

    print("\n================  TOP 5 CONFIGS  ================")
    for i, s in enumerate(top5, 1):
        wE, wP, wV, a, e0, ed, g = s["key"]
        why = [f"final reward={s['final_reward']:.3f}"]
        if s["conv_ep"] is not None and s["conv_ep"] < 1e9:
            why.append(f"convergence ep={s['conv_ep']}")
        else:
            why.append("no convergence")
        print(f"{i}. wE={wE}, wP={wP}, wV={wV} | alpha={a} | eps0={e0} | decay={ed} | gamma={g}")
        print("   -> " + "; ".join(why))

if __name__ == "__main__":
    main()
