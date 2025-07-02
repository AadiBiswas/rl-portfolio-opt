from portfolio_env_execution import ExecutionAwarePortfolioEnv
import numpy as np
from collections import deque

class SharpeRewardExecutionEnv(ExecutionAwarePortfolioEnv):
    """
    Reward = Sharpe-style reward: mean return / std deviation over recent steps.
    Execution-aware: adjusts returns for slippage and transaction cost.
    Includes protections against division by zero, invalid (NaN/inf) values, and exploding rewards.
    Uses a sliding window of recent returns to compute the reward.
    """
    def __init__(self, *args, **kwargs):
        self.sharpe_window = 20  # Window size for recent returns
        self.recent_returns = deque(maxlen=self.sharpe_window)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.recent_returns.clear()
        return super().reset()

    def compute_reward(self, net_return, weights):
        """
        Compute reward based on Sharpe-style ratio: mean / std of recent returns.
        Clipped to prevent reward explosion during training.
        Execution-aware logic adjusts portfolio return before computing reward.
        """
        r = net_return  # Already a scalar

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

        # === Optional: Volatility-aware dynamic weighting (commented out) ===
        # if std < 0.01:
        #     weight = 1.2
        # elif std < 0.02:
        #     weight = 1.0
        # else:
        #     weight = 0.7
        # sharpe_like *= weight

        # === Optional: Gain penalty (commented out) ===
        # if r > 0.05:
        #     penalty = 0.02 * np.log1p(r)
        #     sharpe_like -= penalty

        clipped_reward = np.clip(sharpe_like, -10.0, 10.0)

        if self.verbose and sharpe_like != clipped_reward:
            print(f"[Debug] Raw Sharpe reward {sharpe_like:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

# Alias for external import
PortfolioEnv = SharpeRewardExecutionEnv

