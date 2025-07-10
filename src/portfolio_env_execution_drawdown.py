# === portfolio_env_execution_drawdown.py ===
from portfolio_env_execution import ExecutionAwarePortfolioEnv
import numpy as np
from collections import deque

class DrawdownPenaltyExecutionEnv(ExecutionAwarePortfolioEnv):
    """
    Reward = adaptive blend of log return, Sharpe-like signal, and drawdown penalty
    Execution-aware: adjusts returns for slippage and transaction cost
    - Encourages long-term growth while discouraging large dips
    - Smooth penalty ensures early rewards aren't overly punished
    - Auto-adjusts weights based on volatility context (Sharpe term)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol_window = 20  # rolling window for volatility (Sharpe)
        self.recent_returns = deque(maxlen=self.vol_window)

    def compute_reward(self, net_return, weights):
        # Handle invalid or extreme returns
        r = net_return
        if not np.isfinite(r):
            if self.verbose:
                print(f"[Warning] Non-finite return encountered. Using 0.")
            r = 0.0
        elif r <= -1.0:
            if self.verbose:
                print(f"[Warning] Return too negative for log: {r:.6f}. Clipping to -0.999.")
            r = -0.999

        # === Compute log return (core reward) ===
        log_r = np.log(1 + r)

        # === Sharpe-style term ===
        self.recent_returns.append(r)
        if len(self.recent_returns) >= self.vol_window:
            mean_return = np.mean(self.recent_returns)
            std_dev = np.std(self.recent_returns)
            if not np.isfinite(mean_return) or not np.isfinite(std_dev) or std_dev < 1e-6:
                mean_return, std_dev = 0.0, 1.0
        else:
            mean_return, std_dev = 0.0, 1.0

        sharpe_bonus = (mean_return / std_dev) if std_dev > 1e-6 else 0.0

        # === Drawdown penalty ===
        drawdown = (self.max_portfolio_value - self.portfolio_value) / (self.max_portfolio_value + 1e-8)
        drawdown = np.clip(drawdown, 0.0, 1.0)

        # === Reward scaling based on drawdown severity ===
        if drawdown > 0.05:
            log_weight = 0.65  # downscale reward during deep drawdowns
            drawdown_penalty_weight = 0.60  # stronger penalty
        else:
            log_weight = 0.88
            drawdown_penalty_weight = 0.30

        sharpe_weight = 0.10 if std_dev > 0.015 else 0.0

        # === Gain bonus for full recovery (optional) ===
        gain_bonus = 0.0
        projected_value = self.portfolio_value * (1 + r)
        if projected_value > self.max_portfolio_value:
            rel_gain = (projected_value - self.max_portfolio_value) / (self.max_portfolio_value + 1e-8)
            gain_bonus = 0.1 * np.log1p(rel_gain)
            if self.verbose:
                print(f"[Debug] Recovery bonus applied: {gain_bonus:.6f}")

        # === Final reward formula (drawdown-boosted variant) === MORE AGGRESSIVE THAN NON-EXECUTION VARIANT
        reward = (
            (log_r * 1.3) +                             # Boost log signal
            (sharpe_bonus * sharpe_weight) -            # Sharpe bonus preserved
            (drawdown ** 1.64) * drawdown_penalty_weight + # Heavier penalty on severe dips
            gain_bonus                                  # Reward recovery
        )

        # === Clip for stability ===
        clipped_reward = np.clip(reward, -10.0, 10.0)
        if self.verbose and reward != clipped_reward:
            print(f"[Debug] Raw reward {reward:.6f} clipped to {clipped_reward:.6f}")

        return clipped_reward

# Alias for external import
PortfolioEnv = DrawdownPenaltyExecutionEnv
