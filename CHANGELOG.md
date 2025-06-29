# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2025-06-29
### Added
- Refactored `BasePortfolioEnv` into a fully modular, doctrine-agnostic base class.
  - Introduced abstract `compute_reward()` method to enforce subclass responsibility for reward calculation.
  - Added `seed()` method for reproducibility and consistent training behavior.
  - Incorporated support for:
    - Per-asset weight caps (`max_weight_per_asset`)
    - Drawdown-aware episode control (`allow_drawdown_recovery`)
    - Step count-based termination (`max_episode_length`)
    - Volatility- and history-based reward tracking (`recent_returns`, `portfolio_history`)
  - Portfolio return clipping and invalid weight handling for increased training stability.

- Rewrote `train_agent.py` with CLI arguments for model tagging, rolling window size, and reward doctrine selection.
  - Automatically selects doctrinal environment (Log, Sharpe, Drawdown) based on input args.
  - Training config is fully user-driven; no manual script edits required.

- Rewrote `test_agent.py` to mirror CLI flexibility and doctrinal automation.
  - Auto-loads most recent model matching the `--tag`
  - Accurately reconstructs portfolio trajectory and cumulative returns.
  - Saves evaluation results as tagged `.npy` files under `evaluation/`.

### Changed
- `BasePortfolioEnv` can no longer be trained or evaluated directly.
  - All agents must subclass with a doctrine which implements `compute_reward()`.

### Notes
- LogReturn, SharpeReward, and DrawdownPenalty doctrines are still under construction and may fail to produce positive alpha.
- Execution-aware doctrine variants exist but are not yet validated and excluded from core training/evaluation workflows.
- Rolling window size and test episode length are now fully dynamic and user-defined.

### Next
- Improve doctrine reward functions to achieve consistent alpha across historical test data.
  - **LogReturnEnv**: Improve reward shaping for compounding and positive skew.
  - **SharpeRewardEnv**: Fine-tune reward stability over rolling windows.
  - **DrawdownPenaltyEnv**: Balance capital preservation with upside growth incentives.
- Integrate slippage and transaction cost modeling into stable execution-aware training pipelines.
- Add doctrine-level evaluation metrics and benchmarking scripts.


## [0.3.0] - 2025-06-28
### Added
- Implemented portfolio_env_log_returns.py, portfolio_env_sharpe_ratio.py, and portfolio_env_drawdown.py to introduce rudimentary `LogReturnEnv`, `SharpeRewardEnv`, and `DrawdownPenaltyEnv` doctrine classes which inherit from `BasePortfolioEnv`class.
- Updated `BasePortfolioEnv` class to account for NaN, inf, and zero-division edge cases.

### Notes
- Doctrines are still preliminary and frequently lead to 80-100% portfolio drawdowns.
- `BasePortfolioEnv` can still operate independently of doctrines.
  - Feature will be removed once doctrines demonstrate consistent alpha.
- For the sake of standardised comparison, all models will be trained on:
  - 50k training steps
  - 30-day rolling window
  - 365-day testing window. 

### Next
- Modify doctrines to consistently generate positive alpha.
- Finalize `BasePortfolioEnv` structure to remove potential failure points from multiple inheritance.

## [0.2.2] - 2025-06-27
### Added

- train_agent.py and test_agent.py to establish baseline training and evaluation pipelines using the foundational `BasePortfolioEnv`.

  - train_agent.py: Trains a PPO agent on the core environment using standard SB3 hyperparameters and saves the model to models/.

  - test_agent.py: Evaluates a trained agent on historical price data and reports portfolio performance, reward trajectory, and cumulative return.

- Integrated DummyVecEnv wrapping and model checkpointing for compatibility with stable-baselines3.

### Notes
- These scripts serve as the initial scaffolding for future experiments across doctrine variants.

- Modular and lightweight, they assume a generic reward structure and lack customization for advanced reward shaping (e.g., log return, Sharpe-based consistency, drawdown sensitivity).

### Next
- Finalize and integrate doctrine-specific environment class logic for:

  - LogReturnEnv: Reward scaled by log(1 + return), with optional gain bonus and volatility sensitivity.

  - SharpeRewardEnv: Risk-adjusted reward based on return consistency over a rolling window.

  - DrawdownPenaltyEnv: Penalizes capital depletion and sharp drops while encouraging sustainable growth.



## [0.2.1] - 2025-06-26
### Added

- Modified `BasePortfolioEnv` (PortfolioEnv) within the src/ directory to remain temporarily operable as an RL agent prior to later integration with doctrines like Log Return, Sharpe Reward, and Drawdown Penalty.
  - Basic environment simulating asset returns and portfolio rebalancing.
  - Included rolling window state and daily reward based on raw return. 
  - Lacks reward subclassing, clipping, or drawdown awareness (preserved for pedagogical comparison).
- Uploaded cleaned daily price data (daily_prices.csv) to data/ for reproducible training and evaluation.

### Notes
- Nil

### Next
- Begin development of training and evaluation agents to test operability of Base Portfolio class


## [0.2.0] - 2025-06-25
### Added
- Developed `BasePortfolioEnv`, the shared abstract environment class from which all RL reward models (log-return, Sharpe, drawdown) will inherit.
- Implements complete logic for:
  - Portfolio value simulation and rebalancing
  - Weight clipping and normalization (with optional max weight per asset)
  - Robust action and return validation (NaNs, infinities, extreme crashes)
  - Episode control via `max_episode_length` and depletion-based termination
  - Modular reward delegation to subclasses via `compute_reward()`
  - Rolling window-based state representation for price-based observations
- Verbose debug mode added for step-by-step insight into portfolio behavior, action reliability, and reward structure.

### Notes
- This environment forms the **core control logic** of the project: all reward variants build on this common simulation engine.
- Designed for clarity, reproducibility, and extensibility, enabling clean experimentation with alternative risk-reward tradeoffs in portfolio optimization.

### Next?
- Integrate `BasePortfolioEnv` into a unified registry system
- Add unit tests for `BasePortfolioEnv` simulation and reward delegation


## [0.1.0] - 2025-06-24
### Added
- Initialized project with virtualenv and pip dependencies
- Created notebook for data loading and visualization
- Pulled daily price data for SPY, QQQ, AAPL, GLD, TLT via yfinance
- Plotted cumulative returns and correlation heatmap
- Saved cleaned data (close prices & daily returns) to `data/` as CSVs

### [Next]
- Build custom OpenAI Gym-compatible trading environment
- Implement reward logic and market step simulation
