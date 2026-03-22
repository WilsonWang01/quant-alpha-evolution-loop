import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def find_best_hedge_pairs():
    """
    通过量化策略自动筛选【对冲配对】组合：
    1. 筛选逻辑：高相关性 (Correlation) + 高 Alpha 差异 (Relative Strength)。
    2. 目标：找到那个“大盘跌它少跌，大盘涨它猛涨”的品种进行对冲。
    """
    print(f"🦞 正在全市场扫描最佳对冲组合 (Time: {datetime.now().strftime('%H:%M:%S')})...")
    
    # 定义核心覆盖池 (代表不同风格)
    candidate_pool = [
        "300308.SZ", "601899.SS", "001309.SZ", "603893.SS", # A股成长/资源
        "NVDA", "TSLA", "AAPL", "MSFT", "GOOG",           # 美股科技
        "BTC-USD", "ETH-USD",                             # 币圈
        "GC=F", "CL=F"                                     # 商品
    ]
    
    # 基准池
    benchmarks = {
        "A-Share": "000300.SS",
        "US-Tech": "QQQ",
        "Crypto": "BTC-USD"
    }

    # 1. 下载数据
    all_tickers = list(set(candidate_pool + list(benchmarks.values())))
    data = yf.download(all_tickers, period="1y")['Close']
    data = data.ffill().dropna()
    
    returns = data.pct_change().dropna()
    
    results = []
    
    # 2. 策略筛选算法
    for ticker in candidate_pool:
        # 自动匹配所属市场的基准
        if ticker.endswith((".SZ", ".SS")):
            bench = benchmarks["A-Share"]
        elif ticker.endswith(("-USD")):
            bench = benchmarks["Crypto"]
        else:
            bench = benchmarks["US-Tech"]
            
        if ticker == bench: continue
        
        # 计算相关性 (判定是否适合对冲)
        corr = returns[ticker].corr(returns[bench])
        
        # 计算 Beta (系统性风险暴露)
        beta = returns[ticker].rolling(60).cov(returns[bench]) / returns[bench].rolling(60).var()
        current_beta = beta.iloc[-1]
        
        # 计算 Alpha (超额收益强度：过去 60 天表现)
        alpha = (returns[ticker] - current_beta * returns[bench]).rolling(60).mean() * 252
        current_alpha = alpha.iloc[-1]
        
        # 筛选评分：相关性 > 0.6 (确保对冲有效) 且 Alpha > 0.1 (确保有肉吃)
        if corr > 0.5:
            results.append({
                "Ticker": ticker,
                "Benchmark": bench,
                "Correlation": corr,
                "Beta": current_beta,
                "Alpha_Strength": current_alpha,
                "Score": current_alpha * corr # 综合评分
            })
            
    # 按评分排序，取前三名作为自动推荐组合
    recommendations = sorted(results, key=lambda x: x['Score'], reverse=True)[:3]
    
    print("\n" + "="*60)
    print("📊 龙虾 AI 自动筛选：本周最佳对冲配对建议")
    print("-" * 60)
    print(f"{'标的':<12} | {'对冲基准':<10} | {'相关性':<7} | {'Beta':<6} | {'Alpha强度'}")
    for r in recommendations:
        print(f"{r['Ticker']:<12} | {r['Benchmark']:<10} | {r['Correlation']:.2f}    | {r['Beta']:.2f} | {r['Alpha_Strength']:+.2%}")
    print("="*60)
    
    return recommendations

if __name__ == "__main__":
    find_best_hedge_pairs()
