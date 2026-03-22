import pandas as pd
import numpy as np
import yfinance as yf
import os

# 核心策略库路径
STRATEGY_PATH = "scripts/quant_universal_core.py"

def calculate_universal_metrics(returns, cum_returns):
    """通用绩效评估：核心看 Calmar, Sortino 和 Sharpe"""
    total_days = (returns.index[-1] - returns.index[0]).days
    if total_days == 0: return {}
    annual_ret = (cum_returns.iloc[-1] ** (365 / total_days)) - 1
    peak = cum_returns.expanding().max()
    dd = (cum_returns - peak) / peak
    mdd = dd.min()
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    sortino = (annual_ret / downside_std) if downside_std > 0 else 0
    calmar = (annual_ret / abs(mdd)) if mdd != 0 else 0
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
    return {"Calmar": calmar, "Sortino": sortino, "Sharpe": sharpe, "MDD": mdd, "Annual_Ret": annual_ret}

def backtest_agnostic_momentum(symbol):
    """
    Alpha-V22: 标的无关型自适应动量策略 (Agnostic Adaptive Momentum)
    逻辑：不依赖标的业务，只看其'统计特征'：波动率倒数加权 + 赫斯特指数(Hurst)趋势过滤
    """
    try:
        data = yf.download(symbol, period="5y", progress=False)
        if data.empty: return None
        close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        
        # 1. 趋势强度过滤 (赫斯特指数预估)
        # 简化版逻辑：长短均线多头排列 + 价格位置
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        # 2. 波动率缩减 (Volatility Targeting)
        # 行情稳(波动小)加仓，行情乱(波动大)减仓，不挑标的
        ret = close.pct_change().fillna(0)
        vol = ret.rolling(20).std() * np.sqrt(252)
        target_vol = 0.15 # 目标年化波动率 15%
        
        # 3. 信号生成
        # 只有在牛市基因(>MA200)且处于上升趋势(MA50>MA200)时才入场
        pos = np.zeros(len(close))
        for i in range(200, len(close)):
            if close.iloc[i] > ma200.iloc[i] and ma50.iloc[i] > ma200.iloc[i]:
                # 动态仓位 = 目标波动率 / 当前波动率
                raw_pos = target_vol / vol.iloc[i] if vol.iloc[i] > 0 else 0
                pos[i] = min(1.0, raw_pos) # 最高满仓
            else:
                pos[i] = 0.0 # 趋势走坏，无视标的基本面，直接离场
                
        # 4. 绩效计算 (考虑万三佣金)
        strat_ret = ret.values[1:] * pos[:-1]
        strat_ret -= (np.abs(np.diff(pos)) * 0.0003)
        strat_ret_series = pd.Series(strat_ret, index=close.index[1:])
        cum_strat = (1 + strat_ret_series).cumprod()
        
        return calculate_universal_metrics(strat_ret_series, cum_strat)
    except:
        return None

def run_agnostic_evolution():
    """
    全市场随机抽样验证：只有在不同类型的资产上平均 Calmar > 1.0 才算好策略
    """
    agnostic_pool = [
        "000300.SS", # A股大盘
        "QQQ",        # 美股科技
        "GC=F",       # 黄金
        "BTC-USD",    # 币圈
        "TLT",        # 债市
        "CL=F"        # 原油
    ]
    
    results = []
    print(f"🦞 启动【标的无关】通用策略压力测试...")
    for sym in agnostic_pool:
        m = backtest_agnostic_momentum(sym)
        if m: results.append({"symbol": sym, **m})
    
    avg_calmar = np.mean([r['Calmar'] for r in results])
    print(f"\n📊 跨市场测试结果 (Avg Calmar: {avg_calmar:.2f})")
    for r in results:
        print(f"| {r['symbol']:<10} | Calmar: {r['Calmar']:.2f} | MDD: {r['MDD']:.2%} |")

if __name__ == "__main__":
    run_agnostic_evolution()
