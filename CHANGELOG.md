# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-07-09

### Added
- Pushed latest, parametrically adjusted versions of all doctrine, training, and testing agents
- Uploaded every model tested from this project's genesis
  - **NOTE**: Most models, especially early ones, totally crash their portfolio values
  - Only recent models (generated from updated doctrines) can yeild 20-25% alpha across 365-day testing window

### Next Steps (Feel free to PR!)

- Visualization elements
- New independent doctrines
- New dynamic doctrines

## [0.6.4] - 2025-07-08

### Fixed
- **DynamicDoctrineCallback**:
  - Fixed environment reference bug: switching now correctly targets `.unwrapped` within `DummyVecEnv`.
  - Added forced switch to SHARPE doctrine after warm-up (at step 2000) for runtime validation.
  - Improved debug visibility of doctrine transitions via explicit logging.

- **ExecutionAwareCompositeEnv**:
  - Updated `switch_doctrine()` to reset internal state of the target doctrine (e.g., `recent_returns`), ensuring fair reward reinitialization after each switch.
  - Added debug prints in `step()` and `switch_doctrine()` to trace active doctrine usage throughout training.

### Notes
- Doctrine switching now functions correctly during training. At the warm-up boundary, the SHARPE doctrine is explicitly activated to verify that doctrine routing is respected. 
- The reward routing bug that prevented SHARPE from being invoked has been resolved.
- Average cumulative return across all **composite environments** (execution-aware and non-aware) now ranges between **+20% to +25% over 10k steps**, reflecting effective doctrine switching.
  - **Drawdown penalty** doctrine generally yields the **highest alpha**, benefiting from downside protection.
  - **Log return** doctrine consistently returns lower alpha, especially in execution-aware setups with cost drag.

- We've explicitly avoided overfitting by limiting total training steps (10k per run), staying below overfitting thresholds observed in prior regimes.

### Next
- Optional next-stage enhancements:
  - Begin implementation of **visualization tools** to:
    - Plot portfolio value trajectory and volatility across time
    - Annotate doctrine-switching points
    - Decompose reward components per doctrine
    - Add reward smoothing or soft doctrine blending
    - Visually diagnose overfitting and alpha decay


## [0.6.3] - 2025-07-07

### Optimized
- **SharpeRewardExecutionEnv**:
  - Shortened Sharpe window from 20 to 10 steps for improved reactivity.
  - Reward now multiplies Sharpe ratio by mean return, biasing toward consistent positive gain.
  - Re-enabled volatility-aware weighting to modulate reward under different risk conditions.
  - Preserves compatibility with execution-aware constraints and doctrine purity.

- **LogReturnExecutionEnv**:
  - Enabled previously-commented Sharpe-style bonus for return consistency.
  - Tuned volatility scaling for better alpha under dynamic risk profiles.
  - Calibrated gain bonus logic to better reward portfolio highs without causing reward spikes.
  - Net effect: reward function now generates **positive alpha** even under slippage/transaction cost.

### Notes
- Execution-aware Composite doctrine performance has improved from **-6.4%** to **-4.7%** cumulative return across 10k steps.
- These upgrades reflect our new focus on **doctrinal aggression**—since we now dynamically switch between log, Sharpe, and drawdown strategies, we can afford to be bolder within each.
- Individual doctrines are being refined with alpha optimization in mind, while preserving modularity for future dynamic blending.

### Next
- Continue alpha optimization by:
  - Making dynamic switching more sensitive, especially for mid-trend Sharpe activations.
  - Further refining drawdown reward mechanics for deeper downside control.
  - Experimenting with doctrine-weight blending and soft transitions instead of hard switches.

- Begin implementation of:
  - **Visualization tools** to chart doctrine-switch frequency, portfolio trajectory, and reward decomposition over time.


## [0.6.2] - 2025-07-06

### Fixed
- **train_agent.py**:
  - Corrected doctrinal import paths for `CompositeEnv` and `CompositeExecutionEnv`.
  - Aligned all CLI-triggered reward environments to internal environment aliases.

- **test_agent.py**:
  - Synchronized import logic with training script for smooth evaluation of composite doctrines.
  - Resolved execution-aware composite test failures caused by misaligned naming conventions.

- **dynamic_doctrine_callback.py**:
  - Repaired constructor signature and improved reliability of runtime switching logic.
  - Added verbose logging for runtime doctrine transitions to enhance debug visibility.

### Enhanced
- **portfolio_env_composite.py**:
  - Added `max_episode_length` argument support.
  - Propagated time limit across all doctrine sub-environments.
  - Fixed `step()` compatibility with `test_agent.py` and `DummyVecEnv`.

- **portfolio_env_execution_composite.py**:
  - Same enhancements as non-execution-aware version.
  - Additionally fixed import handling by aligning internal aliases with doctrinal class structure.
  - Ensures compatibility with CLI training/testing pipelines.

### Notes
- Composite (non-execution) achieves **+2.8% cumulative return** over 10k steps using dynamic switching across log, Sharpe, and drawdown doctrines.
- Execution-aware Composite currently yields **-6.4% return**, due to realistic slippage and transaction costs compounding negative alpha in log-return regimes. 
- Doctrine purity has been preserved in both environments by isolating or commenting all cross-doctrinal bonuses (e.g., Sharpe-style consistency in LogReturn).

### Next
- Improve **alpha generation** under both composite environments, especially execution-aware variants, given that this framework most accurately represents the dynamic doctrine adjustment used by quantitative traders.
- Add **visualization tools** to plot reward trajectory, doctrine-switch frequency, and return decomposition over time.
- Explore **doctrine-weight blending** (instead of hard switches), and reward smoothing strategies for more realistic trading simulations.


## [0.6.1] - 2025-07-05

### Enhanced
- **train_agent.py**:
  - Updated doctrinal import logic to support `CompositeEnv` and `CompositeExecutionEnv` through CLI.
  - Allows unified training runs across multiple reward formulations (log, sharpe, drawdown).
  - Execution-aware support preserved via `--execution_aware` flag.
  - Added CLI documentation and training command examples to reflect new doctrine type.

- **test_agent.py**:
  - Added support for evaluating agents trained under composite doctrines.
  - Maintains proper handling for execution-aware variants and internal reward shaping consistency.
  - Extends evaluation logic without disrupting backwards compatibility with legacy environments.

### Notes
- This update brings full CLI compatibility for hybrid doctrine experimentation.
- Composite environments now behave like first-class citizens in both training and testing workflows.

### Next
- Begin live testing of `CompositeEnv` and `DynamicDoctrineCallback`, hyperparameter tuning under dynamic reward conditions to assess overfitting risks and recovery robustness.
- Add plotting + analytics scripts to visualize doctrine-switching behavior and reward component breakdowns.

## [0.6.0] - 2025-07-04

### Added
- **CompositeEnv** (`portfolio_env_composite.py`):
  - Introduces multi-doctrine training using a shared reward interface blending log return, Sharpe ratio, and drawdown penalty.
  - Maintains doctrinal modularity by calling each doctrine's `compute_reward` method individually.

- **Execution-aware CompositeEnv** (`portfolio_env_composite_execution.py`):
  - Extends `CompositeEnv` to support realistic market conditions, incorporating slippage and transaction cost penalties.
  - Enables doctrine blending under execution-aware constraints, preserving doctrine purity while introducing cost realism.

- **DynamicDoctrineCallback** (`dynamic_doctrine_callback.py`):
  - Enables real-time reward doctrine switching during training based on rolling Sharpe ratio performance.
  - Evaluates agent performance every N steps, selecting the doctrine that maximizes risk-adjusted reward.
  - Fully plug-and-play with both execution-aware and standard CompositeEnvs.

### Changed
- **LogReturnExecutionEnv**:
  - Reward structure tuned to approximate break-even returns under slippage and transaction cost conditions.
  - Gain bonuses and Sharpe-style consistency modifiers remain commented to preserve doctrinal integrity.
  - Due to high execution drag and the absence of variance-based shaping, alpha remains difficult to achieve under this reward scheme.

### Notes
- Composite and dynamic doctrine systems represent a foundational step toward regime-switching, multi-objective reinforcement learning.
- All environments preserve reward purity by keeping bonuses or hybridization logic **commented** or **isolated**.
- Dynamic switching logic currently operates via `Callback`; support for full environment-embedded switching may follow.

### Next
- Refactor `train_agent.py` and `test_agent.py` to support `DynamicDoctrineCallback` with minimal user effort.
- Explore weighted doctrine blending and longer performance windows for smoother regime transitions.
- Begin hyperparameter tuning under dynamic reward conditions to assess overfitting risks and recovery robustness.

## [0.5.2] - 2025-07-03

### Added
- **Execution-aware CLI integration for `train_agent.py` and `test_agent.py`**:
  - Added `--execution_aware` flag to both scripts
- **Updated README instructions to account for execution aware doctrines**: 

### Notes
- Nil; Minor update

### Next
- Same as previous

## [0.5.1] - 2025-07-02

### Added
- **Optional enhancements to LogReturnEnv**:
  - Sharpe-style consistency bonus to encourage stable return patterns over time.
  - Training-aware exponential decay to dampen late-stage reward inflation and reduce overfitting.
  - Both features are commented out by default to preserve the “pure” log return doctrine but are documented for advanced users.

- **Optional enhancements to execution-aware LogReturnExecutionEnv**:
  - Sharpe-style consistency bonus and training-aware decay mirrored from the base LogReturnEnv.
  - Extends doctrinal flexibility to environments modeling slippage and transaction costs.
  - Remain commented out to maintain clean separation of concerns.

  - **Optional Volatility-aware dynamic weighting for SharpeRewardEnv**:
  - Adds optional scaling of Sharpe reward based on recent return volatility.
  - Helps stabilize reward signals in low-volatility regimes or suppress excessive spikes in high-volatility environments.
  - Code is commented by default for safe baseline behavior.

- **Optional Volatility-aware weighting in execution-aware SharpeRewardExecutionEnv**:
  - Optional reward modulation based on rolling standard deviation of net returns.
  - Designed to reduce instability in execution-aware training runs.
  - Fully optional and disabled by default.

### Notes
- All new reward shaping logic is **commented by default**, offering toggles for more advanced or robust agent training strategies.
- Enhancements mirror across both base and execution-aware doctrines for consistency.
- This update sets the groundwork for upcoming dynamic doctrine switching and more nuanced reward control.

### Next
- Implement **dynamic doctrine switching** to more realistically simulate shifts in trading regimes or model selection strategies.
- Tune reward function of **LogReturnExecutionEnv**, which currently shows negative alpha under slippage and transaction cost.
- Begin validating hybrid reward schemes combining return, risk, and drawdown sensitivity under realistic execution conditions.


## [0.5.0] - 2025-07-01

### Added
- **ExecutionAwarePortfolioEnv**: New base class extending `BasePortfolioEnv` to support realistic market conditions.
  - Incorporates slippage and transaction cost modeling via per-step weight deltas.
  - Computes execution-adjusted net returns to replace raw portfolio return in reward logic.
  - Fully modular and backwards-compatible with existing doctrine subclasses.

- **Execution-aware LogReturnEnv**: Variant of `LogReturnEnv` that accounts for execution costs.
  - Accepts net return input adjusted for slippage and transaction cost.
  - Retains volatility-aware scaling, gain bonus, and reward clipping.
  - Structurally mirrors base LogReturnEnv but derives from `ExecutionAwarePortfolioEnv`.

- **Execution-aware SharpeRewardEnv**: Cost-aware variant of Sharpe reward formulation.
  - Computes rolling Sharpe-style ratio over net returns.
  - Maintains reward clipping and numerical stability features.
  - Uses a deque-based rolling window for consistent tracking of adjusted return distributions.

- **Execution-aware DrawdownPenaltyEnv**: Execution-aware version of drawdown-sensitive doctrine.
  - Applies log return, Sharpe bonus, and drawdown penalty using net return input.
  - Preserves recovery bonus logic and dynamic reward weighting based on drawdown severity.
  - Aligns structure and reward formulation with its non-execution-aware counterpart.

### Notes
- Execution-aware doctrines now fully implemented and modularized.
- **Execution-aware LogReturnEnv** is currently producing **negative alpha** under slippage—requires further tuning.
- Doctrines are run **independently** and not hybridized (e.g., no Sharpe bonus inside LogReturn or vice versa).
- Execution modeling is structurally sound and fully integrated but still undergoing reward-tuning optimization.

### Next
- **LogReturnEnv**: Introduce optional Sharpe-style consistency bonus to improve long-term stability.
- **SharpeRewardEnv**: Add volatility-aware dynamic weighting (from DrawdownPenaltyEnv) to sharpen reward shaping.
- **Execution-aware LogReturnEnv**: Tune reward structure to overcome negative alpha under transaction friction.
- Explore **hybrid doctrine variants** blending risk-adjusted growth with capital preservation features.


## [0.4.1] - 2025-06-30

### Changed
- **LogReturnEnv**: Improved reward shaping to reflect short-term volatility and long-term performance targets.
  - Added rolling volatility-aware scaling to dynamically adjust reward magnitude.
  - Introduced gain bonus for surpassing previous portfolio highs.
  - Implemented clipping and safety checks to handle numerical edge cases.
  - Updated docstring and added debug logging to support interpretability.

- **SharpeRewardEnv**: Enhanced stability and episodic behavior for risk-adjusted learning.
  - Switched to `deque` for efficient rolling return tracking.
  - Added return and reward clipping to prevent unstable reward spikes.
  - Improved numerical stability via checks for NaNs, infs, and low standard deviation.
  - Added `reset()` method to clear return history between episodes.
  - Expanded docstring and verbosity-controlled debug logging.

- **DrawdownPenaltyEnv**: Rebuilt reward function to balance growth, risk, and recovery incentives.
  - Replaced simple penalized return with composite reward: log return, Sharpe bonus, drawdown penalty, and gain bonus.
  - Dynamically adjusted reward weights based on drawdown severity and portfolio volatility.
  - Applied clipping and validity checks for reward stability.
  - Added recovery bonus to incentivize breakout above historical highs.
  - Enhanced documentation and debug output.

### Notes
- All three doctrine environments now produce **positive cumulative returns** over historical test data.
- **SharpeRewardEnv** and **DrawdownPenaltyEnv** have begun to demonstrate **positive alpha**, offering improved consistency and risk-adjusted growth relative to benchmarks.
- These reward formulations are now more aligned with their respective design principles:  
  - **LogReturn** → compounding and volatility-awareness  
  - **Sharpe** → consistency and variance control  
  - **Drawdown** → resilience and capital preservation

### Next
- Refine **LogReturnEnv** to not only achieve positive returns, but also demonstrate sustained **positive alpha**.
- Begin development of **execution-aware doctrine variants**, incorporating:
  - Slippage modeling
  - Transaction cost penalties
  - Execution-aware state features and reward shaping
- Extend the base environment to support these execution-aware components modularly.


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

### Next
- Integrate `BasePortfolioEnv` into a unified registry system
- Add unit tests for `BasePortfolioEnv` simulation and reward delegation


## [0.1.0] - 2025-06-24
### Added
- Initialized project with virtualenv and pip dependencies
- Created notebook for data loading and visualization
- Pulled daily price data for SPY, QQQ, AAPL, GLD, TLT via yfinance
- Plotted cumulative returns and correlation heatmap
- Saved cleaned data (close prices & daily returns) to `data/` as CSVs

### Next
- Build custom OpenAI Gym-compatible trading environment
- Implement reward logic and market step simulation
