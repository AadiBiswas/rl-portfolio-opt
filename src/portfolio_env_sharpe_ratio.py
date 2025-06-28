# === portfolio_env_sharpe.py ===
from portfolio_env import BasePortfolioEnv
import numpy as np

class SharpeRewardEnv(BasePortfolioEnv):
    """
    Reward = Sharpe-style reward: mean return / std deviation over recent steps.
    Includes protections against division by zero and invalid (NaN/inf) values.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_returns = []
        self.sharpe_window = 20  # You can adjust this if needed

    def compute_reward(self, returns, weights):
        r = np.dot(returns, weights)

        # Ignore invalid return and fallback to 0
        if not np.isfinite(r):
            r = 0.0

        self.recent_returns.append(r)

        # Maintain sliding window
        if len(self.recent_returns) > self.sharpe_window:
            self.recent_returns.pop(0)

        mean = np.mean(self.recent_returns)
        std = np.std(self.recent_returns)

        # Prevent division by zero or invalid values
        if not np.isfinite(mean) or not np.isfinite(std) or std == 0:
            return 0.0

        return mean / std

# Alias class for cleaner import elsewhere
PortfolioEnv = SharpeRewardEnv