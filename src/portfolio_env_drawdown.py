# === portfolio_env_drawdown.py ===
from portfolio_env import BasePortfolioEnv
import numpy as np

class DrawdownPenaltyEnv(BasePortfolioEnv):
    """
    Reward = penalized return: raw return - lambda * drawdown_penalty
    Penalizes large drawdowns relative to historical maximum portfolio value.
    """
    def __init__(self, *args, drawdown_penalty_lambda=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_portfolio_value = self.initial_cash
        self.drawdown_penalty_lambda = drawdown_penalty_lambda

    def compute_reward(self, returns, weights):
        # Compute raw return
        r = np.dot(returns, weights)

        # Ignore invalid return
        if not np.isfinite(r):
            r = 0.0

        # Compute current portfolio value before update
        projected_value = self.portfolio_value * (1 + r)
        self.max_portfolio_value = max(self.max_portfolio_value, projected_value)

        # Drawdown penalty: proportion of drop from max
        drawdown = (self.max_portfolio_value - projected_value) / (self.max_portfolio_value + 1e-8)
        penalty = self.drawdown_penalty_lambda * drawdown

        # Penalized reward
        reward = r - penalty

        # Clip reward to avoid explosion
        return np.clip(reward, -10, 10)

PortfolioEnv = DrawdownPenaltyEnv