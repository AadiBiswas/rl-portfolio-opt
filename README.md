# Reinforcement Learning for Portfolio Optimization

This project applies deep reinforcement learning to the problem of dynamic portfolio allocation in financial markets. Using agents like PPO (Proximal Policy Optimization) and DDPG (Deep Deterministic Policy Gradient), I aim to develop and benchmark strategies across different market conditions and risk settings. This framework is intended to support multiple reward "doctrines"—including Log Return maximization, Sharpe-style risk-adjusted return, and Drawdown-penalized stability.

## Goals
- Simulate and evaluate trading agents using historical market data
- Model transaction costs, slippage, and risk-adjusted returns
- Extend environments and policy networks for practical portfolio management<br><br>

## HOW TO TRAIN AND TEST MODELS IN TERMINAL:
- Step 1: Import relevant data (or test on provided data): `daily_returns.csv`
- Step 2: Activate virtual environment `source venv/bin/activate`
- Step 3: Train model `python train_agent.py --tag run1_window30`
    - NOTE: All models will temporarily be trained on the `BasePortfolioEnv` class, prior to doctrinal implementation
- Step 4: Test model: python test_agent.py (prints your final portfolio value and cumulative returns.)

**🚧 WORK IN PROGRESS**