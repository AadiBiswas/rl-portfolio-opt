# === portfolio_env_log.py ===
class LogReturnEnv(BasePortfolioEnv):
    """Reward = log(1 + portfolio_return)"""
    def compute_reward(self, returns, weights):
        r = np.dot(returns, weights)
        return np.log(1 + r) if r > -1 else -1.0
# Alias for external import
PortfolioEnv = LogReturnEnv