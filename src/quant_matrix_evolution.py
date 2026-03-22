import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import os

# 策略进化实验室日志：knowledge/quant_strategy_matrix.md
MATRIX_LOG = "knowledge/quant_strategy_matrix.md"

def calculate_advanced_metrics(returns, cum_returns):
    """
    计算顶级 Quant 关注的稳定性指标：Calmar, Sortino, Tail Ratio
    """
    if len(returns) < 2: return {}
    total_days = (returns.index[-1] - returns.index[0]).days
    annual_ret = (cum_returns.iloc[-1] ** (365 / max(total_days, 1))) - 1
    
    # MDD
    peak = cum_returns.expanding().max()
    dd = (cum_returns - peak) / peak
    mdd = dd.min()
    
    # Sortino (下行波动风险)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (annual_ret / downside_std) if downside_std > 0 else 0
    
    # Calmar (收益回撤比)
    calmar = (annual_ret / abs(mdd)) if mdd != 0 else 0
    
    # Tail Ratio (95% 分位数与 5% 分位数的比例)
    tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0
    
    return {
        "Annual_Ret": annual_ret,
        "MDD": mdd,
        "Sortino": sortino,
        "Calmar": calmar,
        "Tail_Ratio": tail_ratio
    }

class StrategyEvolutionLab:
    def __init__(self):
        self.best_calmar = 0
        if not os.path.exists("knowledge"): os.makedirs("knowledge")
        if not os.path.exists(MATRIX_LOG):
            with open(MATRIX_LOG, "w") as f:
                f.write("# 🦞 量化策略进化矩阵实验室\n\n| 组合类型 | 策略逻辑 | Calmar | Sortino | MDD | 状态 |\n|---|---|---|---|---|---|\n")

    def run_all_weather_test(self):
        """
        全天候资产对冲策略：股票、黄金、债券的动态对冲
        """
        assets = ["SPY", "TLT", "GLD", "300308.SZ"] # 美股、债、金、A股成长
        print(f"🧪 正在运行【全天候动态对冲】策略变异...")
        
        data = yf.download(assets, period="5y", progress=False)['Close'].ffill().dropna()
        returns = data.pct_change().dropna()
        
        # 变异逻辑：尝试不同的权重分配算法 (目前是 Risk Parity 风险平价)
        vol = returns.rolling(60).std()
        inv_vol = 1.0 / vol
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        
        # 增加动量过滤：只有动量为正的资产才分配权重
        mom = returns.rolling(20).sum()
        weights[mom < 0] = 0
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
        
        port_ret = (returns * weights.shift(1)).sum(axis=1)
        # 扣除调仓损耗
        port_ret -= (weights.diff().abs().sum(axis=1) * 0.0005)
        
        cum_ret = (1 + port_ret).cumprod()
        metrics = calculate_advanced_metrics(port_ret, cum_ret)
        
        self.log_result("全天候对冲", "动量过滤+风险平价", metrics)
        return metrics

    def log_result(self, category, logic, m):
        status = "🔥 接近圣杯" if m['Calmar'] > 2.0 else "✅ 稳定进化"
        with open(MATRIX_LOG, "a") as f:
            f.write(f"| {category} | {logic} | {m['Calmar']:.2f} | {m['Sortino']:.2f} | {m['MDD']:.2%} | {status} |\n")

if __name__ == "__main__":
    lab = StrategyEvolutionLab()
    lab.run_all_weather_test()
