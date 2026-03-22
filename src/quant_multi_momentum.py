import pandas as pd
import numpy as np
import yfinance as yf

def backtest_multi_asset_rotation_v3(tickers):
    """
    🦞 Alpha-V10: 跨品种动量轮动 (稳定版)
    核心思想：不再死磕单一标的。
    每月评估：NVDA/黄金/BTC/A股谁最猛？如果行情都在跌，我就空仓躲雨。
    """
    print(f"🦞 正在执行【Alpha-V10: 多品种动量轮动】策略演习 :: {', '.join(tickers)}")
    
    # 1. 数据下载
    raw_data = yf.download(tickers, period="3y")['Close']
    data = raw_data.ffill().dropna(how='all')
    
    # 2. 动量指标计算 (20日涨幅)
    momentum = data.pct_change(20)
    returns = data.pct_change()
    
    # 3. 策略循环
    strategy_returns = []
    dates = data.index
    
    current_asset = None
    
    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i-1]
        
        # 每月初执行调仓逻辑
        if i == 1 or date.month != prev_date.month:
            moms = momentum.iloc[i-1] # 用前一天的数据做决策，避免未来函数
            valid_moms = moms.dropna()
            
            if not valid_moms.empty:
                best_asset = valid_moms.idxmax()
                # 强度过滤：过去一月涨幅 > 3% 且 绝对值为正
                if valid_moms[best_asset] > 0.03:
                    current_asset = best_asset
                else:
                    current_asset = None
            else:
                current_asset = None
                
        # 计算今日收益
        if current_asset:
            ret = returns.loc[date, current_asset]
            # 扣除滑点/调仓损耗 (每次调仓扣 0.1%)
            if i > 1 and date.month != dates[i-1].month:
                ret -= 0.001
            strategy_returns.append(ret)
        else:
            strategy_returns.append(0.0) # 空仓
            
    # 4. 统计
    strategy_series = pd.Series(strategy_returns, index=dates[1:])
    cum_strat = (1 + strategy_series).cumprod()
    
    # 基准收益 (等权持有)
    bench_returns = returns.mean(axis=1).iloc[1:]
    cum_bench = (1 + bench_returns).cumprod()
    
    print("\n" + "="*50)
    print(f"📊 跨品种轮动报告 (3年) :: {symbol_clean(tickers)}")
    print(f"策略累计收益: {(cum_strat.iloc[-1]-1)*100:.2f}%")
    print(f"基准累计收益: {(cum_bench.iloc[-1]-1)*100:.2f}%")
    print(f"策略最大回撤: {(cum_strat / cum_strat.expanding().max() - 1).min()*100:.2f}%")
    print("="*50)

def symbol_clean(ts):
    return [t.split('.')[0] for t in ts]

if __name__ == "__main__":
    assets = ["NVDA", "GC=F", "BTC-USD", "300308.SZ", "601899.SS"]
    backtest_multi_asset_rotation_v3(assets)
