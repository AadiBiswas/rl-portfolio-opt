import gym
from gym import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    A custom environment for portfolio optimization based on OpenAI Gym.
    
    The agent observes a rolling window of asset prices and outputs portfolio weights.
    Rewards are computed as daily portfolio returns.
    """

    def __init__(self, price_df, initial_cash=1_000_000, window_size=30):
        """
        Initialize the environment.

        Args:
            price_df (pd.DataFrame): DataFrame of historical asset prices (columns = assets).
            initial_cash (float): Starting portfolio value.
            window_size (int): Number of past days to include in each observation.
        """
        super().__init__()

        self.price_df = price_df
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.n_assets = price_df.shape[1]

        # Action space: a portfolio weight for each asset (sum to 1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        # Observation space: price window of shape [window_size, n_assets]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_assets),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.current_step = self.window_size  # Skip the warm-up period
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.done = False

        return self._get_observation()

    def step(self, action):
        """
        Apply portfolio allocation and compute portfolio return.

        Args:
            action (np.ndarray): Portfolio weights (one per asset).

        Returns:
            obs (np.ndarray): Next observation window.
            reward (float): Daily portfolio return.
            done (bool): Whether simulation has ended.
            info (dict): Extra info (empty for now).
        """
        # Ensure valid weights
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights)

        # Compute daily returns
        prev_prices = self.price_df.iloc[self.current_step - 1]
        current_prices = self.price_df.iloc[self.current_step]
        asset_returns = current_prices / prev_prices - 1

        # Portfolio return and update value
        portfolio_return = np.dot(asset_returns.values, weights)
        self.portfolio_value *= (1 + portfolio_return)

        self.current_step += 1
        done = self.current_step >= len(self.price_df) - 1

        reward = portfolio_return
        obs = self._get_observation()

        return obs, reward, done, {}

    def _get_observation(self):
        """
        Get the trailing window of prices used as input to the agent.
        """
        window = self.price_df.iloc[self.current_step - self.window_size : self.current_step]
        return window.values

    def render(self, mode="human"):
        """
        Print current portfolio value.
        """
        print(f"Step: {self.current_step} | Portfolio Value: ${self.portfolio_value:,.2f}")
