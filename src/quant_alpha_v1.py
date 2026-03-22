import pandas as pd
import numpy as np
import yfinance as yf

def backtest_alpha_v16_dynamic_momentum(symbol, name):
    """
    🦞 Alpha-V16: 动态动量轮替 (Dynamic Momentum)
    核心：不再靠“单一趋势”，而是靠“强弱转换”。
    1. 计算 20 日和 60 日两个窗口的动量。
    2. 如果短动量强于长动量（加速上涨），满仓持股。
    3. 如果短动量弱于长动量（减速），减仓至 50%。
    4. 如果双动量转负，全线离场避险。
    这种策略在 A 股的高波动、长震荡环境中，能有效过滤“假金叉”。
    """
    print(f"🦞 正在执行【Alpha-V16: 动态动量轮替】策略回测 :: {name} ({symbol})")
    
    # 1. 数据下载
    raw_data = yf.download(symbol, period="5y")
    if raw_data.empty: return
    close_series = raw_data['Close'].iloc[:, 0] if isinstance(raw_data['Close'], pd.DataFrame) else raw_data['Close']
    close = close_series.values
    
    # 2. 计算动量因子
    mom20 = close_series.pct_change(20).values
    mom60 = close_series.pct_change(60).values
    
    # 3. 仓位逻辑
    pos = np.zeros(len(close))
    
    for i in range(60, len(close)):
        # 如果都在涨，且近期涨幅更快（加速中）
        if mom20[i] > 0 and mom20[i] > mom60[i]:
            pos[i] = 1.0 # 满仓冲锋
        # 如果长线在涨，但短线开始跑不动（减速/震荡）
        elif mom60[i] > 0:
            pos[i] = 0.5 # 半仓防御
        # 只要长线转负，坚决离场
        else:
            pos[i] = 0.0 # 空仓观望
                
    # 4. 绩效核算
    returns = close_series.pct_change().fillna(0).values
    strat_ret = returns[1:] * pos[:-1]
    
    # 扣除损耗
    pos_diff = np.abs(np.diff(pos))
    strat_ret = strat_ret - (pos_diff * 0.0003)
    
    cum_mkt = np.cumprod(1 + returns)
    cum_strat = np.cumprod(1 + strat_ret)

    total_market = (cum_mkt[-1] - 1) * 100
    total_strat = (cum_strat[-1] - 1) * 100
    
    mdd_strat = (cum_strat / np.maximum.accumulate(cum_strat) - 1).min() * 100

    print("\n" + "="*50)
    print(f"📊 {name} Alpha-V16 动态动量报告")
    print(f"基准总收益: {total_market:.2f}% | 策略总收益: {total_strat:.2f}%")
    print(f"超额收益 (Alpha): {total_strat - total_market:+.2f}%")
    print(f"策略最大回撤: {mdd_strat:.2f}%")
    print("="*50)

if __name__ == "__main__":
    targets = [
        ("300308.SZ", "中际旭创"),
        ("NVDA", "英伟达"),
        ("000300.SS", "沪深300")
    ]
    for code, name in targets:
        backtest_alpha_v16_dynamic_momentum(code, name)
