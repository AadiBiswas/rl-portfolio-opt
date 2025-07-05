# === train_agent.py ===
"""
Commands to train per doctrine:

ORIGINAL:
python train_agent.py --tag run_log_return --window 30 --reward_type log 
python train_agent.py --tag run_sharpe --window 30 --reward_type sharpe
python train_agent.py --tag run_drawdown --window 30 --reward_type drawdown
python train_agent.py --tag run_composite --window 30 --reward_type composite

EXECUTION AWARE (slippage + transaction cost):
python train_agent.py --tag run_log_exec --window 30 --reward_type log --execution_aware
python train_agent.py --tag run_sharpe_exec --window 30 --reward_type sharpe --execution_aware
python train_agent.py --tag run_drawdown_exec --window 30 --reward_type drawdown --execution_aware
python train_agent.py --tag run_composite_exec --window 30 --reward_type composite --execution_aware

Adjust window size as desired, e.g. --window 60 for 60-day rolling window.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

# Add 'src' folder to import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# === Parse CLI arguments ===
parser = argparse.ArgumentParser(description="Train PPO agent for portfolio optimization.")
parser.add_argument("--tag", type=str, default="default", help="Custom tag to name the saved model.")
parser.add_argument("--window", type=int, default=30, help="Rolling window size in the environment.")
parser.add_argument("--reward_type", type=str, default="log", choices=["log", "sharpe", "drawdown", "composite"],
                    help="Reward doctrine to use during training.")
parser.add_argument("--execution_aware", action="store_true",
                    help="Use execution-aware environment with slippage and transaction cost.")
args = parser.parse_args()

# === Dynamic reward environment import ===
if args.reward_type == "log":
    if args.execution_aware:
        from portfolio_env_execution_log_return import PortfolioEnv
    else:
        from portfolio_env_log_return import PortfolioEnv
elif args.reward_type == "sharpe":
    if args.execution_aware:
        from portfolio_env_execution_sharpe_ratio import PortfolioEnv
    else:
        from portfolio_env_sharpe_ratio import PortfolioEnv
elif args.reward_type == "drawdown":
    if args.execution_aware:
        from portfolio_env_execution_drawdown import PortfolioEnv
    else:
        from portfolio_env_drawdown import PortfolioEnv
elif args.reward_type == "composite":
    if args.execution_aware:
        from portfolio_env_execution_composite import CompositeExecutionAwareEnv as PortfolioEnv
    else:
        from portfolio_env_composite import CompositeEnv as PortfolioEnv
    from dynamic_doctrine_callback import DynamicDoctrineSwitchCallback
else:
    raise ValueError(f"Unsupported reward type: {args.reward_type}")

# === Load dataset ===
DATA_PATH = os.path.join("data", "daily_prices.csv")
prices_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
assert (prices_df > 0).all().all(), "Non-positive price found. Check input CSV."

# === Set seed for reproducibility ===
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# === Create Gym environment ===
env = DummyVecEnv([
    lambda: PortfolioEnv(price_df=prices_df, window_size=args.window, verbose=True)
])

# === Initialize PPO agent ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.005,
    learning_rate=3e-4,
    clip_range=0.2,
    n_epochs=10,
    seed=42
)

# === Train the agent ===

"""
Modify how many steps you want to train for: 10-50k is good for testing.
At 100k-ish steps, you're at risk of overtraining
overtraining risks: overfitting, hyperaggression, portfolio vals spiking during training but collapsing during eval
"""

print(f"Training model with tag: {args.tag}")

if args.reward_type == "composite":
    callback = DynamicDoctrineSwitchCallback(switch_interval=1500, verbose=True)
    model.learn(total_timesteps=10_000, callback=callback)
else:
    model.learn(total_timesteps=10_000)

# === Save the model ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"ppo_portfolio_{args.tag}_{timestamp}"
save_path = os.path.join("models", model_name)

os.makedirs("models", exist_ok=True)
model.save(save_path)
print(f"Model saved to {save_path}.zip")
