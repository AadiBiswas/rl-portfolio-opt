# === portfolio_env_log.py ===
# Currently generating 2-3% annualized returns (NOTE: No slippage/txn costs)

from portfolio_env import BasePortfolioEnv
import numpy as np
from collections import deque

class LogReturnEnv(BasePortfolioEnv):
    """
    Reward = adaptive log return + Sharpe-style bonus (commented out) + gain bonus
    - Dynamically scales based on volatility context
    - Encourages consistent returns via Sharpe-style term (Optional, commented out)
    - Rewards surpassing prior portfolio highs 
    - Dampens over time to reduce overtraining artifacts (Optional, commented out)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol_window = 30  # Rolling window for volatility context
        self.recent_returns = deque(maxlen=self.vol_window)

    def compute_reward(self, returns, weights):
        # === Portfolio return ===
        r = np.dot(returns, weights)

        # Handle invalid or extreme returns
        if not np.isfinite(r):
            if self.verbose:
                print(f"[Warning] Non-finite return encountered in log reward. Using 0.")
            r = 0.0
        elif r <= -1.0:
            if self.verbose:
                print(f"[Warning] Return too negative for log: {r:.6f}. Clipping to -0.999.")
            r = -0.999

        # === Log return component ===
        log_r = np.log1p(r)

        # Track for volatility-aware scaling
        self.recent_returns.append(r)
        if len(self.recent_returns) >= self.vol_window:
            std_dev = np.std(self.recent_returns)
            scale = 0.4 if std_dev < 0.01 else 0.7 if std_dev < 0.02 else 1.0
        else:
            std_dev = 1.0
            scale = 0.5  # early step default

        reward = log_r * scale  # === Volatility-scaled log return ===

        """""
        # === Sharpe-style consistency bonus ===
        Good to include, but commenting this to preserve "pure" log return doctrine
        sharpe_bonus = 0.0
        if len(self.recent_returns) >= self.vol_window:
            mean_return = np.mean(self.recent_returns)
            sharpe = mean_return / (std_dev + 1e-6)
            sharpe_bonus = 0.03 * sharpe  # dampened
            reward += sharpe_bonus
            if self.verbose:
                print(f"[Debug] Sharpe-style bonus: {sharpe_bonus:.6f}")
        """""
        
        # === Gain bonus for new highs ===
        projected_value = self.portfolio_value * (1 + r)
        gain_bonus = 0.0
        if projected_value > self.max_portfolio_value:
            rel_gain = (projected_value - self.max_portfolio_value) / (self.max_portfolio_value + 1e-8)
            gain_bonus = 0.20 * np.log1p(rel_gain) # adjust parameter based on risk appetite 
            reward += gain_bonus
            if self.verbose:
                print(f"[Debug] Gain bonus applied: {gain_bonus:.6f}")

        # === Training-aware decay (optional) ===
        # decay = 0.995 ** self.steps_elapsed  # exponential decay over steps
        # reward *= decay  # === Dampens late-stage reward inflation ===

        # === Final clipping ===
        clipped_reward = np.clip(reward, -10.0, 10.0)
        if self.verbose and reward != clipped_reward:
            print(f"[Debug] Raw reward {reward:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

# Alias for external import
PortfolioEnv = LogReturnEnv
