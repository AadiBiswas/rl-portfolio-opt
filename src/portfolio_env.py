# === portfolio_env.py ===
import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque

class BasePortfolioEnv(gym.Env):
    """
    Shared base class for portfolio optimization environments.
    Includes logic for action handling, portfolio simulation, and observation.
    Reward logic should be implemented in subclass via `compute_reward`.
    """
    def __init__(self, price_df, initial_cash=1_000_000, window_size=30, verbose=False,
                 max_weight_per_asset=0.5, allow_drawdown_recovery=True, max_episode_length=None):
        super(BasePortfolioEnv, self).__init__()

        self.price_df = price_df
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.verbose = verbose
        self.max_weight_per_asset = max_weight_per_asset
        self.allow_drawdown_recovery = allow_drawdown_recovery
        self.max_episode_length = max_episode_length

        self.n_assets = price_df.shape[1]

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=np.float32
        )

        # Additional tracking for adaptive rewards
        self.vol_window = 20
        self.recent_returns = deque(maxlen=self.vol_window)
        self.portfolio_history = deque(maxlen=50)  # For rolling drawdown
        self.smoothed_reward = 0.0

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        if self.verbose:
            print(f"[Info] Environment seeded with {seed}")

    def reset(self):
        self.current_step = np.random.randint(self.window_size, len(self.price_df) - 1)
        self.portfolio_value = self.initial_cash
        self.max_portfolio_value = self.initial_cash
        self.done = False
        self.steps_elapsed = 0
        self.recent_returns.clear()
        self.portfolio_history.clear()
        self.smoothed_reward = 0.0
        return self._get_observation()

    def step(self, action):
        weights = np.clip(action, 0, 1)
        total_weight = np.sum(weights)

        if not np.isfinite(total_weight) or total_weight == 0:
            if self.verbose:
                print(f"[Warning] Invalid action at step {self.current_step}. Using equal weights.")
            weights = np.ones_like(weights) / self.n_assets
        else:
            weights /= total_weight

        if self.max_weight_per_asset < 1.0:
            if np.any(weights > self.max_weight_per_asset):
                if self.verbose:
                    print(f"[Warning] Allocation cap exceeded at step {self.current_step}. Clipping weights.")
                weights = np.minimum(weights, self.max_weight_per_asset)
                weights /= np.sum(weights)

        prev_prices = self.price_df.iloc[self.current_step - 1]
        current_prices = self.price_df.iloc[self.current_step]
        returns = current_prices / prev_prices - 1

        if not np.all(np.isfinite(returns)):
            if self.verbose:
                print(f"[Warning] Invalid returns at step {self.current_step}. Replacing with 0s.")
            returns = np.nan_to_num(returns)

        if np.any(returns < -0.95):
            if self.verbose:
                print(f"[Warning] Extreme negative return detected at step {self.current_step}. Using equal weights.")
            weights = np.ones_like(weights) / self.n_assets

        if self.verbose:
            print(f"[Debug] Weights: {weights}")
            print(f"[Debug] Returns: {returns}")

        reward = self.compute_reward(returns, weights)

        if not np.isfinite(reward):
            if self.verbose:
                print(f"[Warning] Non-finite reward at step {self.current_step}. Setting to 0.")
            reward = 0.0

        portfolio_return = np.dot(returns, weights)
        if not np.isfinite(portfolio_return):
            if self.verbose:
                print(f"[Warning] Invalid portfolio return at step {self.current_step}. Setting to 0.")
            portfolio_return = 0.0

        portfolio_return = np.clip(portfolio_return, -0.99, 1.0)

        if self.verbose:
            print(f"[Debug] Step {self.current_step} | Return: {portfolio_return:.6f} | Reward: {reward:.6f} | PV before: {self.portfolio_value:.2f}")

        self.portfolio_value *= (1 + portfolio_return)
        self.max_portfolio_value = max(self.portfolio_value, self.max_portfolio_value)

        self.recent_returns.append(portfolio_return)
        self.portfolio_history.append(self.portfolio_value)

        obs = self._get_observation()
        self.current_step += 1
        self.steps_elapsed += 1

        done = (
            self.current_step >= len(self.price_df)
            or self.portfolio_value <= 10.0
            or (self.max_episode_length is not None and self.steps_elapsed >= self.max_episode_length)
        )

        if self.portfolio_value <= 10.0:
            if self.verbose:
                print(f"[Info] Portfolio value nearly depleted.")
            if not self.allow_drawdown_recovery:
                done = True
            else:
                self.portfolio_value = max(self.portfolio_value, 10.0)

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "reward": reward
        }

        return obs, reward, done, info

    def _get_observation(self):
        obs_window = self.price_df.iloc[self.current_step - self.window_size : self.current_step]
        obs = obs_window.values
        if not np.all(np.isfinite(obs)):
            if self.verbose:
                print(f"[Warning] Invalid obs at step {self.current_step}. Replacing with 0s.")
            obs = np.nan_to_num(obs)
        return obs

    def render(self, mode="human"):
        print(f"Step: {self.current_step} | Portfolio Value: ${self.portfolio_value:,.2f}")

    def compute_reward(self, returns, weights):
        raise NotImplementedError("Subclasses must implement reward logic.")
