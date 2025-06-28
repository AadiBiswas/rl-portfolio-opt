import gym
from gym import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    Custom OpenAI Gym environment for portfolio optimization using historical price data.
    Simulates a multi-asset portfolio allocation task with reward based on portfolio returns.
    """
    def __init__(self, price_df, initial_cash=1_000_000, window_size=30, verbose=False):
        super(PortfolioEnv, self).__init__()

        self.price_df = price_df
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.verbose = verbose

        self.n_assets = price_df.shape[1]

        # Action: portfolio weights for each asset (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Observation: historical window of asset prices
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_assets),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Reset environment to initial state.
        """
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.done = False

        return self._get_observation()

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        assert action.shape == (self.n_assets,), f"Invalid action shape: {action.shape}"

        # Normalize and clip action to form portfolio weights
        weights = np.clip(action, 0, 1)
        total_weight = np.sum(weights)

        if not np.isfinite(total_weight) or total_weight == 0:
            if self.verbose:
                print(f"[Warning] Invalid action normalization at step {self.current_step}. Using equal weights.")
            weights = np.ones_like(weights) / self.n_assets
        else:
            weights /= total_weight

        # Get asset prices and compute returns
        prev_prices = self.price_df.iloc[self.current_step - 1]
        current_prices = self.price_df.iloc[self.current_step]
        returns = current_prices / prev_prices - 1

        # Replace any invalid returns
        if not np.all(np.isfinite(returns)):
            if self.verbose:
                print(f"[Warning] Invalid asset returns at step {self.current_step}. Replacing with 0s.")
            returns = np.nan_to_num(returns)

        # Portfolio return
        portfolio_return = np.dot(returns, weights)

        # Clip extreme returns to prevent overflows
        portfolio_return = np.clip(portfolio_return, -1, 1)

        if not np.isfinite(portfolio_return):
            if self.verbose:
                print(f"[Warning] NaN or inf portfolio return at step {self.current_step}. Setting to 0.")
            portfolio_return = 0.0

        self.portfolio_value *= (1 + portfolio_return)

        obs = self._get_observation()
        reward = portfolio_return
        self.current_step += 1
        done = self.current_step >= len(self.price_df)

        if self.verbose:
            print(f"Step: {self.current_step} | Portfolio Value: ${self.portfolio_value:,.2f} | Return: {portfolio_return:.4%}")

        return obs, reward, done, {}

    def _get_observation(self):
        """
        Return the window of historical prices used as observation.
        Replaces any NaNs or infs with zeros to avoid crashing the RL agent.
        """
        obs_window = self.price_df.iloc[self.current_step - self.window_size : self.current_step]
        obs = obs_window.values

        if not np.all(np.isfinite(obs)):
            if self.verbose:
                print(f"[Warning] NaNs/Infs detected in observation at step {self.current_step}. Replacing with 0s.")
            obs = np.nan_to_num(obs)

        return obs

    def render(self, mode="human"):
        """
        Display the current state of the environment.
        """
        print(f"Step: {self.current_step} | Portfolio Value: ${self.portfolio_value:,.2f}")