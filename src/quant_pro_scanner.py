import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(tickers, period="3y"):
    try:
        data = yf.download(tickers, period=period, progress=False)['Close']
        df = data.ffill().dropna(axis=1, how='any')
    except:
        return []

    tickers = df.columns.tolist()
    n = len(tickers)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            s1 = df.iloc[:, i]
            s2 = df.iloc[:, j]
            score, pvalue, _ = coint(s1, s2)
            
            if pvalue < 0.10: # 专业门槛
                X = sm.add_constant(s2)
                model = sm.OLS(s1, X).fit()
                spread = model.resid
                
                spread_lag = spread.shift(1)
                spread_diff = spread.diff()
                df_temp = pd.DataFrame({'diff': spread_diff, 'lag': spread_lag}).dropna()
                if not df_temp.empty:
                    res_ou = sm.OLS(df_temp['diff'], df_temp['lag']).fit()
                    theta = -res_ou.params.iloc[0]
                    half_life = np.log(2) / theta if theta > 0 else 999
                    
                    if half_life < 60: # 必须在3个月内回归
                        pairs.append({
                            "pair": (tickers[i], tickers[j]),
                            "p_value": pvalue,
                            "half_life": half_life,
                            "beta": model.params.iloc[1]
                        })
    return pairs

def run_pro_report():
    print(f"🦞 启动【Alpha-V18: 协整性雷达】专业级重构程序...")
    
    clusters = {
        "A股算力": ["300308.SZ", "300394.SZ", "601138.SS"],
        "A股资源": ["601899.SS", "600547.SS", "600988.SS"],
        "美股科技": ["GOOG", "MSFT", "AAPL", "META"],
        "加密货币": ["BTC-USD", "ETH-USD"]
    }
    
    final_report = "\n# 📊 Alpha-V18 专业级协整配对报告\n\n"
    final_report += "> 筛选逻辑：Engle-Granger 协整检验 (P < 0.1) + OU 过程回归半衰期 (< 60天)。\n\n"
    final_report += "| 板块集群 | 配对品种 | Co-int P值 | 回归半衰期 (天) | 龙虾诊断 |\n"
    final_report += "|---|---|---|---|---|\n"
    
    all_results = []
    for name, tickers in clusters.items():
        pairs = find_cointegrated_pairs(tickers)
        for p in pairs:
            all_results.append({"name": name, **p})
            
    for r in sorted(all_results, key=lambda x: x['p_value']):
        status = "🔥 顶级配对" if r['p_value'] < 0.05 else "✅ 合格配对"
        final_report += f"| {r['name']} | {r['pair'][0]} vs {r['pair'][1]} | {r['p_value']:.4f} | {r['half_life']:.1f} | {status} |\n"
        
    print(final_report)

if __name__ == "__main__":
    run_pro_report()
