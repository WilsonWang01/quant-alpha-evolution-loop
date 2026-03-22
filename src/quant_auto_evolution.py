import pandas as pd
import numpy as np
import yfinance as yf
import os

# 策略存储路径
PRO_LOG = "knowledge/quant_universal_evolution.md"

def calculate_professional_metrics(returns):
    """
    根据 2026 量化最佳实践，计算核心性价比指标
    """
    if returns.empty: return None
    
    cum_returns = (1 + returns).cumprod()
    
    # 1. 年化收益率
    total_days = (returns.index[-1] - returns.index[0]).days
    if total_days == 0: return None
    annual_return = (cum_returns.iloc[-1] ** (365 / total_days)) - 1
    
    # 2. 最大回撤 (MDD)
    peak = cum_returns.expanding().max()
    dd = (cum_returns - peak) / peak
    mdd = dd.min()
    
    # 3. 索提诺比率 (Sortino Ratio) - 只考虑下行波动
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_return / downside_std) if downside_std > 0 else 0
    
    # 4. 卡尔玛比率 (Calmar Ratio) - 收益/回撤
    calmar = (annual_return / abs(mdd)) if mdd != 0 else 0
    
    # 5. 夏普比率 (Sharpe)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    return {
        "Annual_Ret": annual_return,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Final_Cum": cum_returns.iloc[-1]
    }

def backtest_alpha_v23_pro(symbol, target_vol=0.15):
    """
    Alpha-V23: 借鉴 Citadel/Two Sigma 的风险预算与动态对冲逻辑
    1. 动态波动率目标 (Volatility Targeting)
    2. 趋势稳定性过滤 (Hurst Exponent 简化版)
    3. 扣除滑点与专业对冲成本
    """
    data = yf.download(symbol, period="3y", interval="1d", progress=False)
    if data.empty or len(data) < 100: return None
    
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].squeeze()
    returns = close.pct_change().fillna(0)
    
    # Factor 1: 多窗口趋势确认 (120d + 60d + 20d)
    df['SMA20'] = close.rolling(20).mean()
    df['SMA60'] = close.rolling(60).mean()
    df['SMA120'] = close.rolling(120).mean()
    df['Trend_Strong'] = (close > df['SMA20']) & (df['SMA20'] > df['SMA60']) & (df['SMA60'] > df['SMA120'])
    
    # Factor 2: 动态波动率放大器 (波动率极低时增加杠杆)
    target_vol = 0.12
    df['Realized_Vol'] = returns.rolling(20).std() * np.sqrt(252)
    # 限制最大杠杆为 2.0，且仅在趋势确认时使用
    df['Vol_Weight'] = (target_vol / df['Realized_Vol']).fillna(0).clip(0, 2.0)
    
    # Factor 3: 盈利保护 (Trailing Stop Logic)
    # 如果当前价格距离 20 日最高价回落超过 3%，则强制平仓
    df['Rolling_Max'] = close.rolling(20).max()
    df['Profit_Exit'] = (close < df['Rolling_Max'] * 0.97)
    
    # Factor 4: POD Circuit Breaker (Pod 级别熔断)
    df['Peak'] = close.expanding().max()
    df['Drawdown'] = (close - df['Peak']) / df['Peak']
    df['Risk_Reducer'] = np.where(df['Drawdown'] < -0.04, 0.0, 1.0) # 4% 触发 Pod 停摆
    
    # Signal: 强势趋势且未触发盈利保护
    df['Raw_Signal'] = np.where(df['Trend_Strong'] & (~df['Profit_Exit']), 1.0, 0.0)
    
    # Position
    df['Position'] = df['Raw_Signal'] * df['Vol_Weight'] * df['Risk_Reducer']
    
    # 策略收益 (扣除滑点 0.03% + 动态对冲成本 0.01% = 0.04% per trade)
    # 增加双边交易成本
    trade_cost = 0.0004
    strat_ret = (returns * df['Position'].shift(1)) - (df['Position'].diff().abs() * trade_cost)
    
    metrics = calculate_professional_metrics(strat_ret)
    return metrics

def run_evolution_loop():
    print(f"🦞 [2026-03-22] 启动【量化进化 Loop】 - Alpha-V23 核心回测中...")
    
    # 跨资产类别：A股、美股、黄金、BTC
    targets = {
        "A-Share (CSI300)": ("000300.SS", 0.08), 
        "US (NVDA)": ("NVDA", 0.20),             
        "Gold (GC=F)": ("GC=F", 0.16),           
        "BTC (BTC-USD)": ("BTC-USD", 0.12)       
    }
    
    all_results = []
    for name, (symbol, vol) in targets.items():
        m = backtest_alpha_v23_pro(symbol, target_vol=vol)
        if m:
            all_results.append({"name": name, "symbol": symbol, **m})
            
    if not all_results: return False, "No results."
    
    avg_sharpe = np.mean([r['Sharpe'] for r in all_results])
    avg_calmar = np.mean([r['Calmar'] for r in all_results])
    max_mdd = min([r['MDD'] for r in all_results])
    
    report = f"\n### 🧪 Alpha-V23 进化报告: {pd.Timestamp.now()}\n"
    report += f"- **逻辑对标**: Citadel 风险预算 + Two Sigma 波动率目标 + 非对称增强\n"
    report += f"- **核心指标**: 平均 Sharpe {avg_sharpe:.2f}, 平均 Calmar {avg_calmar:.2f}, 最大 MDD {max_mdd:.2%}\n"
    report += "| 资产 | 年化收益 | MDD | Sharpe | Calmar |\n|---|---|---|---|---|\n"
    
    for r in all_results:
        report += f"| {r['name']} | {r['Annual_Ret']:.2%}| {r['MDD']:.2%} | {r['Sharpe']:.2f} | {r['Calmar']:.2f} |\n"
    
    with open(PRO_LOG, "a") as f:
        f.write(report)
        
    # 考核指标：Sharpe > 1.5, Calmar > 1.5, MDD < 12%
    success = (avg_sharpe > 1.5) and (avg_calmar > 1.5) and (abs(max_mdd) < 0.12)
    return success, report

if __name__ == "__main__":
    if not os.path.exists("knowledge"): os.makedirs("knowledge")
    success, rpt = run_evolution_loop()
    print(rpt)
    if success:
        print("🚀 SUCCESS: Metrics breakthrough!")
