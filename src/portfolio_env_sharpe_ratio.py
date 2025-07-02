# === portfolio_env_sharpe.py ===
# Currently generating 15-20% annualized returns (NOTE: No slippage/txn costs)

from portfolio_env import BasePortfolioEnv
import numpy as np
from collections import deque

class SharpeRewardEnv(BasePortfolioEnv):
    """
    Reward = Sharpe-style reward: mean return / std deviation over recent steps.
    Includes protections against division by zero, invalid (NaN/inf) values, and exploding rewards.
    Uses a sliding window of recent returns to compute the reward.
    """
    def __init__(self, *args, **kwargs):
        self.sharpe_window = 20  # Window size for recent returns
        self.recent_returns = deque(maxlen=self.sharpe_window)  # Initialize *before* super().__init__
        super().__init__(*args, **kwargs)

    def reset(self):
        self.recent_returns.clear()  # Clear return history between episodes
        return super().reset()

    def compute_reward(self, returns, weights):
        """
        Compute reward based on Sharpe-style ratio: mean / std of recent returns.
        Clipped to prevent reward explosion during training.
        """
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

        sharpe_like = mean / std

        # === Optional: Volatility-aware dynamic weighting (might dampen model, but good insurance) ===
        # Boost reward in stable environments (low std)
        # if std < 0.01:
        #     weight = 1.2
        # elif std < 0.02:
        #     weight = 1.0
        # else:
        #     weight = 0.7
        # sharpe_like *= weight  # === Amplified & stabilized Sharpe ===

        # === Optional: Gain penalty to discourage unrealistic spikes  (might dampen model, but good insurance)  ===
        # if r > 0.05:
        #     penalty = 0.02 * np.log1p(r)
        #     sharpe_like -= penalty
        #     if self.verbose:
        #         print(f"[Debug] Gain penalty applied: {penalty:.6f}")

        clipped_reward = np.clip(sharpe_like, -10.0, 10.0)

        if self.verbose and sharpe_like != clipped_reward:
            print(f"[Debug] Raw Sharpe reward {sharpe_like:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

# Alias class for cleaner import elsewhere
PortfolioEnv = SharpeRewardEnv