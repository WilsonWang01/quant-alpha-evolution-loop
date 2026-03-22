import pandas as pd
import numpy as np
import yfinance as yf

def backtest_ma_trend_strategy(ticker, short_window=10, long_window=30):
    print(f"🦞 正在拉取 {ticker} 数据并进行【均线趋势追踪】策略回测...")
    
    data = yf.download(ticker, period="2y")
    if data.empty: return None

    # 计算均线
    data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window).mean()

    # 策略逻辑：金叉买入，死叉卖出 (趋势跟随)
    data['Signal'] = 0.0
    
    # 使用 .loc 进行赋值避免警告
    data.loc[data['MA_Short'] > data['MA_Long'], 'Signal'] = 1.0
    
    data['Market_Return'] = data['Close'].pct_change()
    # 策略收益 = 今日涨幅 * 昨日持仓信号
    data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)
    
    data['Cum_Market_Return'] = (1 + data['Market_Return']).cumprod()
    data['Cum_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

    final_market = data['Cum_Market_Return'].iloc[-1]
    final_strategy = data['Cum_Strategy_Return'].iloc[-1]
    
    print("\n" + "="*45)
    print(f"📊 {ticker} 【均线趋势追踪】报告")
    print(f"基准累计收益: {(float(final_market)-1)*100:.2f}%")
    print(f"策略累计收益: {(float(final_strategy)-1)*100:.2f}%")
    print(f"胜过基准: {'✅ 是' if final_strategy > final_market else '❌ 否'}")
    print("="*45)
    
    return data

if __name__ == "__main__":
    target_list = [
        "688012.SS", # 中微公司 (半导体)
        "300308.SZ", # 中际旭创 (AI算力)
        "688256.SS", # 寒武纪 (国产算力)
        "601899.SS"  # 紫金矿业 (避险)
    ]
    
    for stock in target_list:
        backtest_ma_trend_strategy(stock)
