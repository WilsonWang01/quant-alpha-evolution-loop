import pandas as pd
import numpy as np
import yfinance as yf
import os

def calculate_detailed_metrics(returns, cum_returns):
    """计算专业级评估指标"""
    total_days = (returns.index[-1] - returns.index[0]).days
    if total_days == 0: return {}
    annual_ret = (cum_returns.iloc[-1] ** (365 / total_days)) - 1
    peak = cum_returns.expanding().max()
    dd = (cum_returns - peak) / peak
    mdd = dd.min()
    
    # 索提诺比率 (只看下行风险)
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    sortino = (annual_ret / downside_std) if downside_std > 0 else 0
    # 卡尔玛比率 (性价比)
    calmar = (annual_ret / abs(mdd)) if mdd != 0 else 0
    
    return {
        "Annual_Ret": annual_ret,
        "MDD": mdd,
        "Sortino": sortino,
        "Calmar": calmar,
        "Total_Ret": (cum_returns.iloc[-1] - 1)
    }

def backtest_multi_asset_v23(tickers, name):
    """
    Alpha-V23: 资产类别专项优选策略
    逻辑：不强求全能，但在特定资产组合上追求极致稳定性。
    采用“波动率倒数加权(Risk Parity) + 趋势共振”
    """
    print(f"🧪 正在测试资产组合 :: {name} ({', '.join(tickers)})")
    
    try:
        data = yf.download(tickers, period="5y", progress=False)['Close']
        if data.empty: return None
        df = data.ffill().dropna()
        
        returns = df.pct_change().fillna(0)
        
        # 1. 计算各资产权重 (基于 60 日波动率倒数)
        vol = returns.rolling(60).std() * np.sqrt(252)
        inv_vol = 1.0 / vol
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        
        # 2. 趋势过滤 (只有当组合中多数资产站上 MA200 时才入场)
        ma200 = df.rolling(200).mean()
        trend_signal = (df > ma200).sum(axis=1) / len(tickers) > 0.5
        
        # 3. 组合收益
        port_ret = (returns * weights.shift(1)).sum(axis=1)
        # 只有在趋势信号为真时持仓，否则持有现金(0收益)
        strat_ret = port_ret * trend_signal.shift(1).astype(float)
        
        # 扣除调仓损耗 (0.05% 模拟跨资产调仓高损耗)
        strat_ret -= (weights.diff().abs().sum(axis=1) * 0.0005)
        
        cum_strat = (1 + strat_ret.fillna(0)).cumprod()
        
        metrics = calculate_detailed_metrics(strat_ret, cum_strat)
        
        # 提取年度收益
        annual_breakdown = strat_ret.groupby(strat_ret.index.year).apply(lambda x: (1+x).prod()-1)
        
        return {"name": name, "metrics": metrics, "annual": annual_breakdown}
    except Exception as e:
        print(f"❌ {name} 测试失败: {e}")
        return None

def run_asset_class_battle():
    # 定义四个截然不同的“资产战队”
    teams = [
        {"name": "硬核科技对冲组", "assets": ["NVDA", "300308.SZ", "QQQ"]},
        {"name": "全球避险组", "assets": ["GC=F", "TLT", "USDJPY=X"]},
        {"name": "数字黄金组", "assets": ["BTC-USD", "ETH-USD", "MSTR"]},
        {"name": "A股红利组", "assets": ["601288.SS", "600028.SS", "601398.SS"]} # 农行、石化、工行
    ]
    
    all_reports = []
    for team in teams:
        res = backtest_multi_asset_v23(team['assets'], team['name'])
        if res: all_reports.append(res)
        
    print("\n" + "==========================================================")
    print("📊 资产类别“选拔赛”结果汇报 (5年长周期)")
    print("==========================================================")
    print(f"{'资产战队':<15} | {'年化收益':<10} | {'最大回撤':<10} | {'卡尔玛比率'}")
    print("-" * 60)
    for r in all_reports:
        m = r['metrics']
        print(f"{r['name']:<15} | {m['Annual_Ret']:>9.2%} | {m['MDD']:>9.2%} | {m['Calmar']:>10.2f}")
    print("==========================================================")
    
    # 打印年度细节 (针对最稳的那一个)
    best_team = max(all_reports, key=lambda x: x['metrics']['Calmar'])
    print(f"\n🏆 当前最稳战队：{best_team['name']}")
    print("📅 历年收益分布：")
    print(best_team['annual'].map(lambda x: f"{x:.2%}").to_string())

if __name__ == "__main__":
    run_asset_class_battle()
