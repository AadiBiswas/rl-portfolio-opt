# === portfolio_env_execution_sharpe_ratio.py ===

from portfolio_env_execution import ExecutionAwarePortfolioEnv
import numpy as np
from collections import deque

class SharpeRewardExecutionEnv(ExecutionAwarePortfolioEnv):
    """
    Reward = (mean / std) * mean * weight
    - Boosted with longer window and volatility-aware scaling
    - Lower reward scaling under unstable regimes
    """
    def __init__(self, *args, **kwargs):
        self.sharpe_window = 30  # Increased for more smoothing under execution noise
        self.recent_returns = deque(maxlen=self.sharpe_window)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.recent_returns.clear()
        return super().reset()

    def compute_reward(self, net_return, weights):
        r = net_return

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

        sharpe = mean / (std + 1e-6)

        # === Volatility-aware weighting ===
        if std < 0.01:
            weight = 1.2
        elif std < 0.02:
            weight = 1.0
        else:
            weight = 0.6  # Slightly lower than non-exec for stricter penalty

        reward = sharpe * mean * weight

        clipped_reward = np.clip(reward, -10.0, 10.0)
        if self.verbose and reward != clipped_reward:
            print(f"[Debug] Raw Sharpe reward {reward:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

PortfolioEnv = SharpeRewardExecutionEnv
