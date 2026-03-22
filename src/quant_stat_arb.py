import pandas as pd
import numpy as np
import yfinance as yf

def backtest_statistical_arbitrage(ticker_a, ticker_b):
    """
    🦞 Alpha-V7: 统计套利 + 配对交易 (Statistical Arbitrage)
    寻找两只高度相关的股票（如半导体同板块），利用其价差回归规律套利。
    这种策略不依赖标的单边上涨，只要两者的'默契'还在就能赚钱。
    """
    print(f"🦞 正在执行【Alpha-V7: 配对交易】策略演习 :: {ticker_a} & {ticker_b}")
    
    # 1. 获取数据
    data = yf.download([ticker_a, ticker_b], period="2y")['Close']
    if data.empty: return
    
    # 2. 计算价差 (Spread) 和 Z-Score (标准化偏离度)
    # 计算比率 A/B
    data['Ratio'] = data[ticker_a] / data[ticker_b]
    
    # 计算均值和标准差 (30日滚动)
    data['MA_Ratio'] = data['Ratio'].rolling(window=30).mean()
    data['Std_Ratio'] = data['Ratio'].rolling(window=30).std()
    
    # Z-Score 表示当前比率偏离均线多少个标准差
    data['Z_Score'] = (data['Ratio'] - data['MA_Ratio']) / data['Std_Ratio']
    
    # 3. 交易逻辑：均值回归
    # 当 Z-Score > 1.5: 比率太高，卖出 A 买入 B (预期比率下行)
    # 当 Z-Score < -1.5: 比率太低，买入 A 卖出 B (预期比率上行)
    # 当 Z-Score 回到 0 附近: 平仓
    
    data['Pos_A'] = 0.0
    data['Pos_B'] = 0.0
    
    position = 0 # 0: 空仓, 1: 做多Ratio, -1: 做空Ratio
    
    pos_a_list = []
    pos_b_list = []
    
    for i in range(len(data)):
        z = data['Z_Score'].iloc[i]
        if np.isnan(z):
            pos_a_list.append(0.0)
            pos_b_list.append(0.0)
            continue
            
        if position == 0:
            if z < -1.5: # 严重偏低，买 A 卖 B
                position = 1
            elif z > 1.5: # 严重偏高，卖 A 买 B
                position = -1
        elif position == 1:
            if z >= 0: # 回归均值，平仓
                position = 0
        elif position == -1:
            if z <= 0: # 回归均值，平仓
                position = 0
                
        if position == 1:
            pos_a_list.append(1.0)
            pos_b_list.append(-1.0)
        elif position == -1:
            pos_a_list.append(-1.0)
            pos_b_list.append(1.0)
        else:
            pos_a_list.append(0.0)
            pos_b_list.append(0.0)
            
    data['Pos_A'] = pos_a_list
    data['Pos_B'] = pos_b_list
    
    # 4. 计算收益
    data['Ret_A'] = data[ticker_a].pct_change()
    data['Ret_B'] = data[ticker_b].pct_change()
    
    # 组合收益 = A收益*昨日持仓 + B收益*昨日持仓
    data['Strategy_Ret'] = data['Ret_A'] * data['Pos_A'].shift(1) + data['Ret_B'] * data['Pos_B'].shift(1)
    
    # 简单基准：50/50 持有
    data['Benchmark_Ret'] = 0.5 * data['Ret_A'] + 0.5 * data['Ret_B']
    
    cum_strat = (1 + data['Strategy_Ret'].fillna(0)).cumprod()
    cum_bench = (1 + data['Benchmark_Ret'].fillna(0)).cumprod()
    
    print("\n" + "="*50)
    print(f"📊 配对交易回测报告: {ticker_a} vs {ticker_b}")
    print(f"策略累计收益: {(cum_strat.iloc[-1]-1)*100:.2f}%")
    print(f"基准累计收益: {(cum_bench.iloc[-1]-1)*100:.2f}%")
    print(f"最大回撤: {(cum_strat / cum_strat.expanding().max() - 1).min()*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    # 挑选同行业高相关性品种
    # 1. A股半导体封测两龙头：长电科技 vs 通富微电
    # 2. 算力 CPO 两剑客：中际旭创 vs 天孚通信
    pairs = [
        ("600584.SS", "002156.SZ"), # 长电 vs 通富
        ("300308.SZ", "300394.SZ")  # 旭创 vs 天孚
    ]
    for a, b in pairs:
        try:
            backtest_statistical_arbitrage(a, b)
        except Exception as e:
            print(f"❌ 对 {a}/{b} 回测失败: {e}")
