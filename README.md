# Reinforcement Learning for Portfolio Optimization 

**VERSION 1 COMPLETE!**

## Overview  
This project applies deep reinforcement learning to the problem of dynamic portfolio allocation in financial markets. Using agents like **PPO** (Proximal Policy Optimization) and **DDPG** (Deep Deterministic Policy Gradient), I aim to develop and benchmark strategies across different market conditions and risk settings.

## Reward Doctrines Supported  
- **Log Return** maximization  
- **Sharpe-style** risk-adjusted return  
- **Drawdown-penalized** stability  
- **Composite mode**: dynamically alternates between doctrines based on context

## Performance  
On average, trained agents achieve **20–25% alpha** over 10K steps, even in **execution-aware** regimes that factor in slippage and transaction costs. Doctrines that penalize drawdowns often outperform pure log-return objectives under realistic conditions.

## Training Philosophy  
Training durations are **intentionally constrained** to avoid overfitting. Aggression is only encouraged when coupled with **dynamic doctrine switching**, allowing the agent to hedge its behavior depending on the risk context.

## Assets Included  
All model checkpoints are uploaded for **full transparency**, including:
- Early training failures  
- Overfit model variants  
- Final candidate strategies



## Goals
- Simulate and evaluate trading agents using historical market data
- Model txn costs, slippage, and risk-adjusted returns
- Extend environments and policy networks for practical portfolio management<br><br>

## HOW TO TRAIN AND TEST MODELS IN TERMINAL:
- **Step 1**: Import relevant data (or test on provided data): `data/daily_prices.csv`  
  *(Note: I now implemented price data, not precomputed returns)*

- **Step 2**: Activate your virtual environment  
  ```bash
  source venv/bin/activate
  ```

- **Step 3**: Train a model by selecting the doctrine and rolling window size  
  ```bash
  python train_agent.py --tag run_log_return --window 30 --reward_type log
  python train_agent.py --tag run_sharpe --window 30 --reward_type sharpe
  python train_agent.py --tag run_drawdown --window 30 --reward_type drawdown
  python train_agent.py --tag run_composite --window 30 --reward_type composite
  ```

  For execution-aware variants

    ```bash
  python train_agent.py --tag run_log_exec --window 30 --reward_type log --execution_aware
  python train_agent.py --tag run_sharpe_exec --window 30 --reward_type sharpe --execution_aware
  python train_agent.py --tag run_drawdown_exec --window 30 --reward_type drawdown --execution_aware
  python train_agent.py --tag run_composite_exec --window 30 --reward_type composite --execution_aware
  ```

- **Step 4**: Test the trained model (ensure tag, window, and reward_type match training config)  
  ```bash
  python test_agent.py --tag run_log_return --window 30 --length 365 --reward_type log
  python test_agent.py --tag run_sharpe --window 30 --length 365 --reward_type sharpe
  python test_agent.py --tag run_drawdown --window 30 --length 365 --reward_type drawdown
  python test_agent.py --tag run_composite --window 30 --length 365 --reward_type composite
  ```
  For execution-aware variants

    ```bash
  python test_agent.py --tag run_log_exec --window 30 --length 365 --reward_type log --execution_aware
  python test_agent.py --tag run_sharpe_exec --window 30 --length 365 --reward_type sharpe --execution_aware
  python test_agent.py --tag run_drawdown_exec --window 30 --length 365 --reward_type drawdown --execution_aware
  python test_agent.py --tag run_composite_exec --window 30 --length 365 --reward_type composite --execution_aware
  ```

> **Tip**: Adjust `--length` to control test horizon (e.g. `--length 365` for one year).  
> Ensure the `--window` used in testing matches the one used during training.
