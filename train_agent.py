import os
import sys 
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 1. Load data
DATA_PATH = os.path.join("data", "daily_returns.csv")
returns_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# 2. Instantiate environment
env = DummyVecEnv([
    lambda: PortfolioEnv(price_df=returns_df, window_size=30, verbose=False)
])

# 3. Instantiate agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.0,
    learning_rate=3e-4,
    clip_range=0.2,
    n_epochs=10,
)

# 4. Train agent
model.learn(total_timesteps=100_000)

# 5. Save model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_portfolio")
print("Model saved to models/ppo_portfolio.zip")