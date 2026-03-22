import pandas as pd
import numpy as np
import yfinance as yf

def backtest_ma_trend_strategy(ticker, short_window=10, long_window=30):
    print(f"🦞 正在拉取 {ticker} 数据并进行【均线趋势追踪】策略回测...")
    
    # 扩大数据获取范围至 5 年，以确保更稳健
    data = yf.download(ticker, period="5y")
    if data.empty: 
        print(f"❌ {ticker} 数据拉取为空")
        return None

    # 计算均线
    data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window).mean()

    # 策略逻辑：金叉买入，死叉卖出 (趋势跟随)
    data['Signal'] = 0.0
    data.loc[data['MA_Short'] > data['MA_Long'], 'Signal'] = 1.0
    
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)
    
    data['Cum_Market_Return'] = (1 + data['Market_Return']).cumprod()
    data['Cum_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()

    final_market = data['Cum_Market_Return'].iloc[-1]
    final_strategy = data['Cum_Strategy_Return'].iloc[-1]
    
    # 计算年化收益率
    days = (data.index[-1] - data.index[0]).days
    annual_market = (float(final_market) ** (365/days) - 1) * 100
    annual_strategy = (float(final_strategy) ** (365/days) - 1) * 100
    
    print("\n" + "="*45)
    print(f"📊 {ticker} 【趋势追踪】长期报告 (5年)")
    print(f"基准总收益: {(float(final_market)-1)*100:.2f}% (年化: {annual_market:.2f}%)")
    print(f"策略总收益: {(float(final_strategy)-1)*100:.2f}% (年化: {annual_strategy:.2f}%)")
    print(f"胜过基准: {'✅ 是' if final_strategy > final_market else '❌ 否'}")
    print("="*45)
    
    return data

if __name__ == "__main__":
    # 挑选不同市场的品种进行横向对比
    target_list = [
        "NVDA",      # 英伟达 (美股科技龙头)
        "GC=F",      # 黄金期货 (避险资产)
        "BTC-USD",   # 比特币 (高波动数字货币)
        "300308.SZ"  # 中际旭创 (A股对照组)
    ]
    
    for stock in target_list:
        try:
            backtest_ma_trend_strategy(stock)
        except Exception as e:
            print(f"❌ {stock} 回测失败: {e}")
