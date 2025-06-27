# test_agent.py
import os
import sys
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# (0) Local import for custom environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from portfolio_env import PortfolioEnv

# (1) Load test data
DATA_PATH = os.path.join("data", "daily_returns.csv")  # Replace with test set if needed
returns_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# (2) Recreate environment (must match training config)
env = DummyVecEnv([
    lambda: PortfolioEnv(price_df=returns_df, window_size=30, verbose=False)
])

# (3) Load trained model
MODEL_PATH = os.path.join("models", "ppo_portfolio.zip")
model = PPO.load(MODEL_PATH, env=env)
print(f"Loaded model from {MODEL_PATH}")

# (4) Evaluate model
obs = env.reset()
done = False
total_reward = 0
portfolio_values = []

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]
    
    # Extract portfolio value (from internal env)
    portfolio_val = env.get_attr("portfolio_value")[0]
    portfolio_values.append(portfolio_val)

# (5) Summary statistics
print(f"\n--- Evaluation Summary ---")
print(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
print(f"Cumulative return: {((portfolio_values[-1] / portfolio_values[0]) - 1) * 100:.2f}%")
print(f"Total reward (sum of log returns): {total_reward:.4f}")

# (Optional) Save portfolio trajectory for later analysis/plotting
np.save("evaluation/portfolio_values.npy", np.array(portfolio_values))
print("Saved portfolio trajectory to evaluation/portfolio_values.npy")