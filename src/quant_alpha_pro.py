import pandas as pd
import numpy as np
import yfinance as yf

def calculate_mdd(cum_returns):
    """计算最大回撤"""
    return (cum_returns / cum_returns.expanding().max() - 1).min() * 100

def backtest_alpha_v16_pro(symbol, name):
    print(f"🦞 正在执行【Alpha-V16 Pro: 深度绩效评估】:: {name} ({symbol})")
    
    # 1. 下载数据
    raw_data = yf.download(symbol, period="5y")
    if raw_data.empty: return
    close_series = raw_data['Close'].iloc[:, 0] if isinstance(raw_data['Close'], pd.DataFrame) else raw_data['Close']
    
    # 2. 策略逻辑 (V16 动态动量)
    mom20 = close_series.pct_change(20).values
    mom60 = close_series.pct_change(60).values
    close_vals = close_series.values
    
    pos = np.zeros(len(close_vals))
    for i in range(60, len(close_vals)):
        if mom20[i] > 0 and mom20[i] > mom60[i]:
            pos[i] = 1.0 # 满仓加速
        elif mom60[i] > 0:
            pos[i] = 0.5 # 半仓震荡
        else:
            pos[i] = 0.0 # 空仓避险
            
    # 3. 计算收益
    returns = close_series.pct_change().fillna(0)
    strat_ret = (returns.values[1:] * pos[:-1]) - (np.abs(np.diff(pos)) * 0.0003)
    
    # 转换为 Series 方便按时间重采样
    strat_ret_series = pd.Series(strat_ret, index=close_series.index[1:])
    mkt_ret_series = returns.iloc[1:]
    
    cum_strat = (1 + strat_ret_series).cumprod()
    cum_mkt = (1 + mkt_ret_series).cumprod()

    # 4. 统计年度收益率
    annual_strat = strat_ret_series.groupby(strat_ret_series.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    annual_mkt = mkt_ret_series.groupby(mkt_ret_series.index.year).apply(lambda x: (1 + x).prod() - 1) * 100
    
    # 5. 核心指标汇总
    total_ret = (cum_strat.iloc[-1] - 1) * 100
    mdd = calculate_mdd(cum_strat)
    vol = strat_ret_series.std() * np.sqrt(252) * 100
    sharpe = (strat_ret_series.mean() * 252) / (strat_ret_series.std() * np.sqrt(252)) if strat_ret_series.std() != 0 else 0

    print("\n" + "="*55)
    print(f"📊 {name} ({symbol}) 策略深度绩效表")
    print("-" * 55)
    print(f"总累计收益: {total_ret:.2f}%")
    print(f"最大回撤: {mdd:.2f}%")
    print(f"年化波动率: {vol:.2f}%")
    print(f"夏普比率 (Sharpe): {sharpe:.2f}")
    print("-" * 55)
    print("🗓️ 年度收益率对照 (Strategy vs Market):")
    comparison = pd.DataFrame({'Strategy (%)': annual_strat, 'Market (%)': annual_mkt})
    print(comparison.round(2).to_string())
    print("="*55)

if __name__ == "__main__":
    targets = [
        ("300308.SZ", "中际旭创"),
        ("NVDA", "英伟达"),
        ("000300.SS", "沪深300")
    ]
    for code, name in targets:
        backtest_alpha_v16_pro(code, name)
