# RL_project
Reinforcement Learning Exploration Strategies in MiniGrid
This project implements and evaluates multiple ε-greedy exploration strategies for tabular Q-learning agents in MiniGrid environments.
It focuses on comparing fixed decay, novelty-based, entropy-driven, and count-based exploration methods, measuring their impact on learning speed, state coverage, and final performance.

## Features
Four exploration strategies:

- Decay – Standard ε decay over time.
- Novelty – Adaptive ε using normalized entropy, learning progress, and variance.
- Entropy – Adaptive ε based on Q-value distribution entropy.
- Count-Based – Intrinsic reward bonus for visiting less-explored states.

## Tested environments:
- MiniGrid-DoorKey-6x6-v0
- MiniGrid-Unlock-v0

## Evaluation metrics:
- Average reward
- Success rate
- Unique states visited
- Heatmaps of Q-values and state visits

## Project Structure

RL_project/
│── agent/                     # Q-learning agent and exploration policies
│   ├── q_learning_agent.py
│   ├── policies.py
│   └── __init__.py
│
│── utils/                      # Helper functions for env setup, saving, rewards
│   ├── env_utils.py
│   ├── reward_shaping.py
│   ├── save_utils.py
│   └── state_utils.py
│── plots/                      # Plots from training for 10 seeds and 4 strategies
│── results/                    # Plots and CSVs for evaluation
│   ├── eval/                   # Evaluation summaries
│   └── plots/                  # Average reward & success rate plots
│
│── tuning/                     # Hyperparameter tuning results
│── q_tables/                   # Saved Q-tables by strategy and seed
│── main.py                     # Main training entry point
│── evaluation.py               # Evaluation script
│── tune_novelty.py              # Novelty tuning script
│── config.yaml                  # Default configuration
│── baseline_config.yaml         # Default Configuration used later when Tuning
│── tuning_config.yaml           # Tuning configuration
│── requirements.txt             # Python dependencies


