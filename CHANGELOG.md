# Changelog

All notable changes to this project will be documented in this file.

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
- Integrate `BasePortfolioEnv` into a unified registry system for dynamic reward switching (e.g., CLI flag for `--reward_type`)
- Add unit tests for `BasePortfolioEnv` simulation and reward delegation
- Finalize reward modules for drawdown and Sharpe variants with improved stability

## [0.1.0] - 2025-06-24
### Added
- Initialized project with virtualenv and pip dependencies
- Created notebook for data loading and visualization
- Pulled daily price data for SPY, QQQ, AAPL, GLD, TLT via yfinance
- Plotted cumulative returns and correlation heatmap
- Saved cleaned data (close prices & daily returns) to `data/` as CSVs
