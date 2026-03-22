import pandas as pd
import numpy as np
import yfinance as yf
import os

# 策略日志
HEDGE_LOG = "knowledge/quant_hedge_evolution.md"

def backtest_market_neutral_strategy(symbol, hedge_symbol="000300.SS", period="5y"):
    """
    🦞 Alpha-V17: 市场中性对冲策略 (Market Neutral Hedge)
    核心逻辑：做多强势标的，同时做空（或对冲）大盘。
    目标：消除市场系统性风险（Beta），赚取纯粹的超额收益（Alpha）。
    """
    print(f"🦞 正在执行【Alpha-V17: 市场中性对冲】策略演习 :: {symbol} vs {hedge_symbol}")
    
    # 1. 获取数据
    data = yf.download([symbol, hedge_symbol], period=period, progress=False)['Close']
    if data.empty: return None
    
    df = data.ffill().dropna()
    
    # 2. 计算收益率
    df['Ret_S'] = df[symbol].pct_change()
    df['Ret_H'] = df[hedge_symbol].pct_change()
    
    # 3. 动态对冲系数 (Beta 匹配)
    # 使用滚动窗口计算标相对于大盘的 Beta 值
    rolling_window = 60
    covariance = df['Ret_S'].rolling(rolling_window).cov(df['Ret_H'])
    variance = df['Ret_H'].rolling(rolling_window).var()
    df['Beta'] = covariance / variance
    
    # 4. 策略逻辑：
    # 只有当标的本身处于短期强势（MA20上方）时才入场，并按 Beta 对冲大盘
    df['MA20'] = df[symbol].rolling(20).mean()
    df['Signal'] = np.where(df[symbol] > df['MA20'], 1.0, 0.0)
    
    # 对冲后的收益 = 标的收益 - (Beta * 大盘收益)
    # 这里模拟的是“完全中性”：做多 1 份标的，做空 Beta 份大盘
    df['Hedged_Ret'] = (df['Ret_S'] - df['Beta'] * df['Ret_H']) * df['Signal'].shift(1)
    
    # 5. 绩效统计
    # 考虑对冲端的成本（如借券费、保证金损耗等，模拟年化 2% 损耗）
    cost_per_day = 0.02 / 252
    df['Strategy_Ret'] = df['Hedged_Ret'] - (df['Signal'].shift(1) * cost_per_day)
    
    cum_strat = (1 + df['Strategy_Ret'].fillna(0)).cumprod()
    cum_mkt = (1 + df['Ret_S'].fillna(0)).cumprod()
    
    total_ret = (cum_strat.iloc[-1] - 1)
    total_mkt = (cum_mkt.iloc[-1] - 1)
    mdd = (cum_strat / cum_strat.expanding().max() - 1).min()
    
    # 计算夏普（对冲策略看重的是夏普，而不是绝对涨幅）
    sharpe = (df['Strategy_Ret'].mean() * 252) / (df['Strategy_Ret'].std() * np.sqrt(252)) if df['Strategy_Ret'].std() != 0 else 0
    
    return {
        "symbol": symbol,
        "hedge": hedge_symbol,
        "ret": total_ret,
        "mkt_ret": total_mkt,
        "mdd": mdd,
        "sharpe": sharpe
    }

def run_hedge_experiment():
    # 测试不同属性的标的
    pairs = [
        ("300308.SZ", "000300.SS"), # 科技龙头 vs 沪深300
        ("601899.SS", "000300.SS"), # 避险资源 vs 沪深300
        ("NVDA", "QQQ")              # 英伟达 vs 纳指
    ]
    
    results = []
    for s, h in pairs:
        res = backtest_market_neutral_strategy(s, h)
        if res: results.append(res)
        
    report = f"\n### 🛡️ 对冲策略实验轮次: {pd.Timestamp.now()}\n"
    report += "| 标的 | 对冲标的 | 对冲后收益 | 标的原收益 | 夏普比率 | 最大回撤 |\n|---|---|---|---|---|---|\n"
    for r in results:
        report += f"| {r['symbol']} | {r['hedge']} | {r['ret']:.2%} | {r['mkt_ret']:.2%} | {r['sharpe']:.2f} | {r['mdd']:.2%} |\n"
        
    with open(HEDGE_LOG, "a") as f:
        f.write(report)
    return report

if __name__ == "__main__":
    if not os.path.exists("knowledge"): os.makedirs("knowledge")
    print(run_hedge_experiment())
