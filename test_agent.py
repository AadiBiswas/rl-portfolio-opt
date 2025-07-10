# === test_agent.py ===
"""
Commands to test per doctrine:

ORIGINAL:
python test_agent.py --tag run_log_return --window 30 --length 365 --reward_type log 
python test_agent.py --tag run_sharpe --window 30 --length 365 --reward_type sharpe
python test_agent.py --tag run_drawdown --window 30 --length 365 --reward_type drawdown
python test_agent.py --tag run_composite --window 30 --length 365 --reward_type composite

EXECUTION AWARE (slippage + transaction cost):
python test_agent.py --tag run_log_exec --window 30 --length 365 --reward_type log --execution_aware
python test_agent.py --tag run_sharpe_exec --window 30 --length 365 --reward_type sharpe --execution_aware
python test_agent.py --tag run_drawdown_exec --window 30 --length 365 --reward_type drawdown --execution_aware
python test_agent.py --tag run_composite_exec --window 30 --length 365 --reward_type composite --execution_aware

Ensure window size matches training config, e.g. --window 30 for 30-day rolling window.
Adjust test length as desired, e.g. --length 365 for 1 year of evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# (0) Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, required=True, help="Run tag used during training")
parser.add_argument("--window", type=int, default=30, help="Rolling window size used during training")
parser.add_argument("--length", type=int, default=None, help="Optional max number of steps during evaluation")
parser.add_argument("--reward_type", type=str, default="log", choices=["log", "sharpe", "drawdown", "composite"],
                    help="Reward type used during training (affects how portfolio value is calculated)")
parser.add_argument("--execution_aware", action="store_true",
                    help="Use execution-aware environment with slippage and transaction costs")
args = parser.parse_args()

# (1) Local import for custom environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# === Dynamic import based on reward type and execution-awareness ===
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
        from portfolio_env_execution_composite import ExecutionAwareCompositeEnv as PortfolioEnv
    else:
        from portfolio_env_composite import CompositeEnv as PortfolioEnv
else:
    raise ValueError(f"Unsupported reward type: {args.reward_type}")

# (2) Load test price data (not returns)
DATA_PATH = os.path.join("data", "daily_prices.csv")
prices_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# (3) Recreate environment (must match training config)
env = DummyVecEnv([
    lambda: PortfolioEnv(price_df=prices_df, window_size=args.window, verbose=True, max_episode_length=args.length)
])

# (4) Locate latest model matching tag
model_dir = "models"
matching_models = sorted([
    f for f in os.listdir(model_dir)
    if f.startswith(f"ppo_portfolio_{args.tag}")
], reverse=True)

if not matching_models:
    raise FileNotFoundError(f"No model found with tag '{args.tag}' in models/")

model_path = os.path.join(model_dir, matching_models[0])
model = PPO.load(model_path, env=env)
print(f"Loaded model from {model_path}")

# (5) Evaluate model
obs = env.reset()
done = False
total_reward = 0
portfolio_values = [1_000_000]  # Starting capital
step_count = 0

while not done and (args.length is None or step_count < args.length):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

    if args.reward_type == "sharpe" or args.reward_type == "composite":
        # Use actual portfolio value from env info dict
        current_value = info[0]["portfolio_value"] if isinstance(info, list) else info["portfolio_value"]
    else:
        # For log and drawdown: use reward to derive new portfolio value
        linear_return = np.exp(reward[0]) - 1
        current_value = portfolio_values[-1] * (1 + linear_return)

    portfolio_values.append(current_value)
    step_count += 1

# (6) Summary statistics
print(f"\n--- Evaluation Summary ---")
print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
print(f"Cumulative return: {((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%")
print(f"Total reward (sum of log returns): {total_reward:.4f}")

# (7) Save results
os.makedirs("evaluation", exist_ok=True)
np.save(f"evaluation/portfolio_values_{args.tag}.npy", np.array(portfolio_values))
print(f"Saved portfolio trajectory to evaluation/portfolio_values_{args.tag}.npy")
