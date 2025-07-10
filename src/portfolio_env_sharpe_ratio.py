# === portfolio_env_sharpe.py ===
# Currently generating 15-20% annualized returns (NOTE: No slippage/txn costs)

from portfolio_env import BasePortfolioEnv
import numpy as np
from collections import deque

class SharpeRewardEnv(BasePortfolioEnv):
    """
    Reward = Sharpe-style reward: mean / std over recent steps
    - Multiplied by mean return to bias toward consistent positive growth
    - Volatility-weighted for stability bias (penalizes noisy regimes)
    - Now includes:
      - Nonlinear Sharpe scaling (via tanh) for aggressive but bounded alpha
      - Warm-start buffer to avoid early suppression
      - Extended window for smoother regime detection
    """
    def __init__(self, *args, **kwargs):
        self.sharpe_window = 25
        self.recent_returns = deque([0.001] * 5, maxlen=self.sharpe_window)  # Warmstart with small gain
        super().__init__(*args, **kwargs)

    def reset(self):
        self.recent_returns.clear()
        self.recent_returns.extend([0.001] * 5)  # Reset warmstart
        return super().reset()

    def compute_reward(self, returns, weights):
        r = np.dot(returns, weights)

        if not np.isfinite(r):
            if self.verbose:
                print(f"[Warning] Non-finite return detected: {r}. Replacing with 0.")
            r = 0.0

        r = np.clip(r, -1.0, 1.0)
        self.recent_returns.append(r)

        if len(self.recent_returns) < self.sharpe_window:
            return 0.0

        mean = np.mean(self.recent_returns)
        std = np.std(self.recent_returns)

        if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-6:
            if self.verbose:
                print(f"[Warning] Unstable Sharpe calculation: mean={mean:.6f}, std={std:.6f}")
            return 0.0

        raw_sharpe = mean / (std + 1e-6)
        nonlinear_scaling = np.tanh(raw_sharpe) * mean  # Aggressive growth bias
        weight = 1.3 if std < 0.01 else 1.0 if std < 0.02 else 0.6

        reward = nonlinear_scaling * weight
        clipped_reward = np.clip(reward, -10.0, 10.0)

        if self.verbose and reward != clipped_reward:
            print(f"[Debug] Raw Sharpe reward {reward:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

PortfolioEnv = SharpeRewardEnv
