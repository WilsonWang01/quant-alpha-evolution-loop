import pandas as pd
import numpy as np
import yfinance as yf
from multiprocessing import Pool, cpu_count
import time
import os

def single_stock_backtest(ticker):
    """
    单个标的的回测逻辑（被并行调用）
    """
    try:
        # 使用较短时间范围进行压力测试
        data = yf.download(ticker, period="3y", progress=False)
        if data.empty: return None
        
        close = data['Close'].iloc[:, 0] if isinstance(data['Close'], pd.DataFrame) else data['Close']
        
        # 简单趋势策略用于演示并行速度
        ma20 = close.rolling(20).mean()
        signal = np.where(close > ma20, 1.0, 0.0)
        
        returns = close.pct_change().fillna(0)
        strat_ret = returns.values[1:] * signal[:-1]
        
        cum_ret = np.prod(1 + strat_ret) - 1
        return {"ticker": ticker, "ret": cum_ret}
    except:
        return None

def run_parallel_backtest(ticker_list):
    """
    并行回测主引擎
    """
    cores = cpu_count()
    print(f"🚀 启动并行回测引擎 :: 监测到 {cores} 个核心")
    print(f"📡 任务总量: {len(ticker_list)} 个标的")
    
    start_time = time.time()
    
    # 开启进程池
    with Pool(processes=cores) as pool:
        results = pool.map(single_stock_backtest, ticker_list)
    
    end_time = time.time()
    
    valid_results = [r for r in results if r is not None]
    print(f"\n✅ 并行回测完成！")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 成功回测标的数: {len(valid_results)}")
    
    # 按收益排序展示前 5
    top_5 = sorted(valid_results, key=lambda x: x['ret'], reverse=True)[:5]
    print("\n🏆 策略表现前 5 名:")
    for res in top_5:
        print(f"- {res['ticker']}: {res['ret']:.2%}")

if __name__ == "__main__":
    # 模拟一个较大的股票池
    sample_pool = [
        "300308.SZ", "300394.SZ", "601138.SS", "688012.SS", "600584.SS",
        "NVDA", "AAPL", "MSFT", "GOOG", "AMD", "TSLA", "META",
        "BTC-USD", "ETH-USD", "GC=F", "TLT", "QQQ", "SPY"
    ] * 2 # 复制一份模拟 36 个任务
    
    run_parallel_backtest(sample_pool)
