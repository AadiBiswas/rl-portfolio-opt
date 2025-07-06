# === portfolio_env_composite.py ===
import gym
import numpy as np
from portfolio_env_log_return import LogReturnEnv
from portfolio_env_sharpe_ratio import SharpeRewardEnv
from portfolio_env_drawdown import DrawdownPenaltyEnv

class CompositeEnv(gym.Env):
    """
    Composite non-execution-aware environment that dynamically switches between
    log return, Sharpe ratio, and drawdown doctrines at runtime.
    Accepts `max_episode_length` and passes it to each underlying doctrine.
    """
    def __init__(self, price_df, window_size=30, verbose=False, max_episode_length=None):
        super().__init__()
        self.verbose = verbose
        self.envs = {
            "log": LogReturnEnv(price_df=price_df, window_size=window_size,
                                verbose=verbose, max_episode_length=max_episode_length),
            "sharpe": SharpeRewardEnv(price_df=price_df, window_size=window_size,
                                      verbose=verbose, max_episode_length=max_episode_length),
            "drawdown": DrawdownPenaltyEnv(price_df=price_df, window_size=window_size,
                                           verbose=verbose, max_episode_length=max_episode_length)
        }
        self.current_env = self.envs["log"]  # default

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
PortfolioEnv = CompositeEnv
