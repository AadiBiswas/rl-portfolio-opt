# === portfolio_env_execution.py ===
import gym
from gym import spaces
import numpy as np
import pandas as pd
from collections import deque

class ExecutionAwarePortfolioEnv(gym.Env):
    """
    Extension of BasePortfolioEnv that includes slippage and transaction cost modeling.
    Execution-aware mechanics are injected just before portfolio value update.
    Reward logic remains delegated to subclasses via `compute_reward()`.
    """
    def __init__(self, price_df, initial_cash=1_000_000, window_size=30, verbose=False,
                 max_weight_per_asset=0.5, allow_drawdown_recovery=True, max_episode_length=None,
                 slippage_rate=0.001, transaction_cost_rate=0.002):
        super().__init__()

        self.price_df = price_df
        self.initial_cash = initial_cash
        self.window_size = window_size
        self.verbose = verbose
        self.max_weight_per_asset = max_weight_per_asset
        self.allow_drawdown_recovery = allow_drawdown_recovery
        self.max_episode_length = max_episode_length

        self.slippage_rate = slippage_rate
        self.transaction_cost_rate = transaction_cost_rate

        self.n_assets = price_df.shape[1]

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.n_assets), dtype=np.float32
        )

        self.recent_returns = deque(maxlen=20)
        self.portfolio_history = deque(maxlen=50)

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
        self.previous_weights = np.zeros(self.n_assets)  # Assume all-cash starting allocation
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
                print(f"[Warning] Extreme negative return detected. Using equal weights.")
            weights = np.ones_like(weights) / self.n_assets

        raw_return = np.dot(returns, weights)
        if not np.isfinite(raw_return):
            raw_return = 0.0

        raw_return = np.clip(raw_return, -0.99, 1.0)

        # === Apply slippage and transaction costs ===
        execution_cost = self._compute_execution_cost(weights)
        net_return = raw_return - execution_cost

        # ✅ Only this line changed!
        reward = self.compute_reward(net_return, weights)

        if not np.isfinite(reward):
            reward = 0.0

        if self.verbose:
            print(f"[Debug] Raw return: {raw_return:.6f}, Execution cost: {execution_cost:.6f}, Net return: {net_return:.6f}")

        self.portfolio_value *= (1 + net_return)
        self.max_portfolio_value = max(self.portfolio_value, self.max_portfolio_value)

        self.recent_returns.append(net_return)
        self.portfolio_history.append(self.portfolio_value)
        self.previous_weights = weights.copy()

        obs = self._get_observation()
        self.current_step += 1
        self.steps_elapsed += 1

        done = (
            self.current_step >= len(self.price_df)
            or self.portfolio_value <= 10.0
            or (self.max_episode_length is not None and self.steps_elapsed >= self.max_episode_length)
        )

        if self.portfolio_value <= 10.0:
            if not self.allow_drawdown_recovery:
                done = True
            else:
                self.portfolio_value = max(self.portfolio_value, 10.0)

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "reward": reward,
            "execution_cost": execution_cost
        }

        return obs, reward, done, info

    def _get_observation(self):
        obs_window = self.price_df.iloc[self.current_step - self.window_size : self.current_step]
        obs = obs_window.values
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs)
        return obs

    def render(self, mode="human"):
        print(f"Step: {self.current_step} | Portfolio Value: ${self.portfolio_value:,.2f}")

    def compute_reward(self, returns, weights):
        raise NotImplementedError("Subclasses must implement reward logic.")

    def _compute_execution_cost(self, new_weights, old_weights=None):
        """
        Compute transaction cost as slippage proportional to the change in portfolio weights.
        You can replace 0.001 with your desired slippage rate (e.g., 0.1% per unit turnover).
        """
        if old_weights is None:
            old_weights = self.previous_weights if hasattr(self, 'previous_weights') else np.zeros_like(new_weights)

        weight_change = np.abs(new_weights - old_weights)
        slippage = self.slippage_rate * np.sum(np.abs(new_weights))  # Based on total allocation
        transaction = self.transaction_cost_rate * np.sum(weight_change)
        return slippage + transaction
