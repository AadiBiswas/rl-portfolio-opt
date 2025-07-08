# === portfolio_env_execution_composite.py ===


import gym
import numpy as np
from portfolio_env_execution_log_return import LogReturnEnv
from portfolio_env_execution_sharpe_ratio import SharpeRewardExecutionEnv
from portfolio_env_execution_drawdown import DrawdownPenaltyExecutionEnv

class ExecutionAwareCompositeEnv(gym.Env):
    """
    Composite execution-aware environment that switches between doctrines dynamically.
    Includes slippage and transaction cost adjustments.
    """
    def __init__(self, price_df, window_size=30, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.envs = {
            "log": LogReturnEnv(price_df=price_df, window_size=window_size, verbose=verbose),
            "sharpe": SharpeRewardExecutionEnv(price_df=price_df, window_size=window_size, verbose=verbose),
            "drawdown": DrawdownPenaltyExecutionEnv(price_df=price_df, window_size=window_size, verbose=verbose)
        }
        self.current_env = self.envs["log"]

        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def switch_doctrine(self, doctrine):
        if doctrine in self.envs:
            self.current_env = self.envs[doctrine]
            if self.verbose:
                print(f"[Switch] Doctrine switched to: {doctrine}")

    def reset(self):
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode="human"):
        return self.current_env.render(mode=mode)

    def seed(self, seed=None):
        for env in self.envs.values():
            env.seed(seed)

# Alias
PortfolioEnv = ExecutionAwareCompositeEnv
