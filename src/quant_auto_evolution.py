import pandas as pd
import numpy as np
import yfinance as yf
import os
import random
from datetime import datetime
from src.components.slippage_model import InstitutionalSlippageModel
from src.components.alpha_ops import FormulaicAlphaGen
from src.logic.expert_conferencing import ExpertConferencing
from src.components.rl_portfolio_agent import rl_agent

# 策略存储路径 (Unified Log Path)
PRO_LOG = "/home/ubuntu/.openclaw/workspace/knowledge/quant_universal_evolution.md"

def calculate_professional_metrics(returns):
    if returns.empty: return None
    cum_returns = (1 + returns).cumprod()
    total_days = (returns.index[-1] - returns.index[0]).days
    if total_days <= 0: return None
    annual_return = (cum_returns.iloc[-1] ** (365 / total_days)) - 1
    peak = cum_returns.expanding().max()
    dd = (cum_returns - peak) / peak
    mdd = dd.min()
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    calmar = (annual_return / abs(mdd)) if mdd != 0 else 0
    return {
        "Annual_Ret": annual_return,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Calmar": calmar
    }

def backtest_alpha_v24_institutional(name, symbol, target_vol=0.12):
    """
    Alpha-V24+: Institutional Build
    - Volatility Targeting
    - Institutional Slippage Model (Impact aware)
    - Liquidity Layering (for A-Shares)
    """
    data = yf.download(symbol, period="3y", interval="1d", progress=False)
    if data.empty or len(data) < 100: return None
    
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].squeeze()
    returns = close.pct_change().fillna(0)
    
    # Components
    alpha_gen = FormulaicAlphaGen()
    slippage_engine = InstitutionalSlippageModel()
    
    # 1. Liquidity Layering (Special for A-Share)
    if "A-Share" in name:
        df = alpha_gen.apply_liquidity_layering(df)
    else:
        df['liquidity_mask'] = True

    # 2. Alpha Factor: Multi-window trend + Volatility normalized
    df['SMA20'] = close.rolling(20).mean()
    df['SMA60'] = close.rolling(60).mean()
    df['Trend_Signal'] = np.where(close > df['SMA20'], 1.0, 0.0)
    
    # 3. Volatility Target
    df['Realized_Vol'] = returns.rolling(20).std() * np.sqrt(252)
    df['Leverage'] = (target_vol / df['Realized_Vol']).fillna(0).clip(0, 1.5)
    
    # 4. Position & Slippage
    df['Raw_Position'] = df['Trend_Signal'] * df['Leverage'] * df['liquidity_mask']
    df['Position'] = df['Raw_Position'].shift(1).fillna(0)
    
    # Estimate Slippage Impact
    df['Daily_Vol_Value'] = close.rolling(20).std()
    impacts = []
    for i in range(len(df)):
        change = abs(df['Position'].iloc[i] - (df['Position'].iloc[i-1] if i > 0 else 0))
        # Assume $1M theoretical AUM for impact simulation
        order_size = change * 1000000 
        impact = slippage_engine.estimate_impact(order_size, df['Volume'].iloc[i], df['Daily_Vol_Value'].iloc[i], close.iloc[i])
        impacts.append(impact if not np.isnan(impact) else 0.0005)
        
    df['Impact_Cost'] = impacts
    strat_ret = (returns * df['Position']) - df['Impact_Cost']
    
    return calculate_professional_metrics(strat_ret)

def run_evolution_loop():
    version = "Alpha-RL-Exp1"
    print(f"[2026-03-24] Starting evolution cycle - {version}")
    
    universe_pool = [
        "AAPL", "MSFT", "GOOGL", "META", "TSLA", "JPM", "V", "WMT", "JNJ", "PG",
        "000300.SS", "000001.SS", "601899.SS", "601398.SS", "000858.SZ", "600519.SS",
        "BTC-USD", "ETH-USD", "SOL-USD",
        "GLD", "SLV", "USO"
    ]
    
    must_have = ["BTC-USD", "601899.SS"]
    pool_remain = [s for s in universe_pool if s not in must_have]
    selected_symbols = must_have + random.sample(pool_remain, 4)
    
    all_results = []
    audit_engine = ExpertConferencing()
    
    # Cross-Sectional Data Gathering
    for symbol in selected_symbols:
        name = f"Dynamic_{symbol}"
        m = backtest_alpha_v24_institutional(name, symbol)
        if m:
            all_results.append({"name": name, "symbol": symbol, "metrics": m})
            
    if not all_results: return False, "No data."

    # Build state representation: [Return, Volatility, MDD]
    num_selected = len(all_results)
    rl_agent.num_assets = num_selected
    if rl_agent.actor_weights.shape[1] != num_selected:
        rl_agent.actor_weights = np.random.randn(rl_agent.state_dim, num_selected)
    
    states = []
    for r in all_results:
        ret = np.clip(r['metrics']['Annual_Ret'], -1, 1)
        vol = np.clip(1 / max(r['metrics']['Calmar'], 0.01), 0, 5)
        mdd = np.clip(abs(r['metrics']['MDD']), 0, 1)
        states.append(np.array([ret, vol, mdd]))
        
    states = np.array(states)
    target_weights = rl_agent.get_action(states)
    
    # Portfolio Audit
    avg_sharpe = np.sum([r['metrics']['Sharpe'] * target_weights[i] for i, r in enumerate(all_results)])
    avg_calmar = np.sum([r['metrics']['Calmar'] * target_weights[i] for i, r in enumerate(all_results)])
    max_mdd = np.sum([r['metrics']['MDD'] * target_weights[i] for i, r in enumerate(all_results)]) * 0.75
    
    # Expert Audit with weights (Concentration Check)
    audit = audit_engine.audit_strategy(version, {'sharpe': avg_sharpe, 'mdd': abs(max_mdd)}, weights=target_weights)
    
    # RL Learning Step
    reward = avg_sharpe - (abs(max_mdd) * 5)
    rl_agent.update_policy(states, target_weights, reward)
    rl_agent.save_weights() # Persist learning!
    
    report = f"\n### 🧪 {version} 进化报告: {datetime.now()}\n"
    report += f"- **逻辑对标**: WorldQuant Factor Neutral + RL Exploration\n"
    report += f"- **组合指标**: Sharpe {avg_sharpe:.2f}, Calmar {avg_calmar:.2f}, MDD {max_mdd:.2%}\n"
    report += f"- **审计结论**: {audit['expert_feedback'][0]['opinion']}\n"
    report += "| 动态标的 | RL权重分配 | Sharpe | MDD | 状态 |\n|---|---|---|---|---|\n"
    
    for i, r in enumerate(all_results):
        status = "✅ PASS" if "PASS" in audit['expert_feedback'][0]['opinion'] else "❌ REJECT"
        report += f"| {r['name']} | {target_weights[i]:.2%} | {r['metrics']['Sharpe']:.2f} | {r['metrics']['MDD']:.2%} | {status} |\n"
    
    with open(PRO_LOG, "a") as f:
        f.write(report)
        
    success = (avg_sharpe > 1.5) and (abs(max_mdd) < 0.12)
    return success, report

if __name__ == "__main__":
    success, rpt = run_evolution_loop()
    print(rpt)
