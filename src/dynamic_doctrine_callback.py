# === dynamic_doctrine_callback.py ===

from stable_baselines3.common.callbacks import BaseCallback

class DynamicDoctrineSwitchCallback(BaseCallback):
    """
    Custom callback to dynamically switch reward doctrines based on portfolio performance.
    
    Args:
        switch_interval (int): Number of steps between doctrine evaluations.
        verbose (int): Verbosity level (0 = silent, 1+ = log changes).
    """
    def __init__(self, switch_interval=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = switch_interval
        self.performance_window = []
        self.current_doctrine = "log"

    def _on_step(self) -> bool:
        # Only evaluate every `eval_freq` steps
        if self.n_calls % self.eval_freq != 0:
            return True

        # Check that we can safely call .env.envs[0]
        env = self.training_env.envs[0]
        current_value = getattr(env, 'portfolio_value', None)
        if current_value is None:
            return True

        self.performance_window.append(current_value)
        if len(self.performance_window) > 5:
            self.performance_window.pop(0)

        # Compare trend to switch doctrine
        if len(self.performance_window) == 5:
            deltas = [j - i for i, j in zip(self.performance_window[:-1], self.performance_window[1:])]
            avg_change = sum(deltas) / len(deltas)

            if avg_change < 0:
                new_doctrine = "drawdown"
            elif avg_change > 1000:
                new_doctrine = "sharpe"
            else:
                new_doctrine = "log"

            if new_doctrine != self.current_doctrine:
                env.switch_doctrine(new_doctrine)
                if self.verbose:
                    print(f"[Callback] Doctrine switched to: {new_doctrine.upper()} based on avg delta {avg_change:.2f}")
                self.current_doctrine = new_doctrine

        return True
