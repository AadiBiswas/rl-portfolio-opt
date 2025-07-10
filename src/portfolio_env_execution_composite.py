# === portfolio_env_execution_composite.py ===
# Added max_episode_length as an additional support argument to the constructor, Propagated it to all three doctrine envs, fixed type error, and ensured compatibility between train and test agents


import gym
import numpy as np
from portfolio_env_execution_log_return import LogReturnExecutionEnv
from portfolio_env_execution_sharpe_ratio import SharpeRewardExecutionEnv
from portfolio_env_execution_drawdown import DrawdownPenaltyExecutionEnv

class ExecutionAwareCompositeEnv(gym.Env):
    """
    Composite execution-aware environment that switches between doctrines dynamically.
    Includes slippage and transaction cost adjustments.
    Accepts `max_episode_length` and passes it to each underlying doctrine.
    """
    def __init__(self, price_df, window_size=30, verbose=False, max_episode_length=None):
        super().__init__()
        self.verbose = verbose
        self.envs = {
            "log": LogReturnExecutionEnv(price_df=price_df, window_size=window_size,
                                verbose=verbose, max_episode_length=max_episode_length),
            "sharpe": SharpeRewardExecutionEnv(price_df=price_df, window_size=window_size,
                                               verbose=verbose, max_episode_length=max_episode_length),
            "drawdown": DrawdownPenaltyExecutionEnv(price_df=price_df, window_size=window_size,
                                                    verbose=verbose, max_episode_length=max_episode_length)
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
