# Quant Strategy Evolution Loop

A quantitative strategy evolution and backtesting engine for multi-asset classes, designed to simulate professional-grade portfolio management (Citadel-style risk budgeting + Two Sigma volatility targeting).

## Core Components

- **Backtesting Engine**: `src/quant_alpha_v1.py`, `src/quant_backtest_multi.py`, etc.
- **Auto-Evolution**: `src/quant_auto_evolution.py` (The main "Alpha Evolution Loop" script).
- **Risk Management**: `src/quant_risk_engine.py` (Implementation of volatility and drawdown limits).
- **Watchdog**: `scripts/quant_watchdog.sh` (Health check and automatic recovery for the background loop).

## Strategy Performance Targets (Institutional Grade)

- **Sharpe Ratio**: > 1.5
- **Calmar Ratio**: > 1.5
- **Max Drawdown**: < 12%

## Asset Classes

- A-Shares (CSI300)
- US Tech (NVDA, etc.)
- Commodities (Gold)
- Crypto (BTC)

## Current Status

The system is in continuous "evolution mode," refining factor directions based on historical performance and risk-adjusted returns.
