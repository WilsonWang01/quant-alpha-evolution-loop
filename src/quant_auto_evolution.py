import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
from src.components.slippage_model import InstitutionalSlippageModel
from src.components.alpha_ops import FormulaicAlphaGen
from src.logic.expert_conferencing import ExpertConferencing

# 策略存储路径
PRO_LOG = "knowledge/quant_universal_evolution.md"

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
        impact = slippage_engine.estimate_impact(order_size, df['Volume'].iloc[i], df['Daily_Vol_Value'].iloc[i])
        impacts.append(impact if not np.isnan(impact) else 0.0005)
        
    df['Impact_Cost'] = impacts
    strat_ret = (returns * df['Position']) - df['Impact_Cost']
    
    return calculate_professional_metrics(strat_ret)

def run_evolution_loop():
    version = "Alpha-V24+ (Institutional Build)"
    print(f"🦞 [2026-03-22] 启动【量化进化 Loop】 - {version} 核心回测中...")
    
    targets = {
        "A-Share (CSI300)": "000300.SS",
        "US (NVDA)": "NVDA",
        "Gold (GC=F)": "GC=F",
        "BTC (BTC-USD)": "BTC-USD"
    }
    
    all_results = []
    audit_engine = ExpertConferencing()
    
    for name, symbol in targets.items():
        m = backtest_alpha_v24_institutional(name, symbol)
        if m:
            # Audit Step
            audit = audit_engine.audit_strategy(name, {'sharpe': m['Sharpe'], 'mdd': abs(m['MDD'])})
            all_results.append({"name": name, "symbol": symbol, "metrics": m, "audit": audit})
            
    if not all_results: return False, "No results."
    
    avg_sharpe = np.mean([r['metrics']['Sharpe'] for r in all_results])
    avg_calmar = np.mean([r['metrics']['Calmar'] for r in all_results])
    max_mdd = min([r['metrics']['MDD'] for r in all_results])
    
    report = f"\n### 🧪 {version} 进化报告: {datetime.now()}\n"
    report += f"- **逻辑对标**: Citadel 风险预算 + Two Sigma 波动率目标 + 机构级滑点模拟\n"
    report += f"- **核心优化**: 引入 A股 流动性分层 (Top 30%) + 专家会诊审计机制\n"
    report += f"- **核心指标**: 平均 Sharpe {avg_sharpe:.2f}, 平均 Calmar {avg_calmar:.2f}, 最大 MDD {max_mdd:.2%}\n"
    report += "| 资产 | Sharpe | Calmar | MDD | 专家审计 |\n|---|---|---|---|---|\n"
    
    for r in all_results:
        status = "✅ PASS" if "PASS" in r['audit']['expert_feedback'][0]['opinion'] else "❌ REJECT"
        report += f"| {r['name']} | {r['metrics']['Sharpe']:.2f} | {r['metrics']['Calmar']:.2f} | {r['metrics']['MDD']:.2%} | {status} |\n"
    
    with open(PRO_LOG, "a") as f:
        f.write(report)
        
    success = (avg_sharpe > 2.0) and (avg_calmar > 2.5) and (abs(max_mdd) < 0.08)
    return success, report

if __name__ == "__main__":
    if not os.path.exists("knowledge"): os.makedirs("knowledge")
    success, rpt = run_evolution_loop()
    print(rpt)
    if success:
        print("🚀 SUCCESS: Institutional Metrics breakthrough!")
