import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from multiprocessing import Pool, cpu_count
import itertools
import time

def test_pair_cointegration(args):
    """
    单个配对的协整检验逻辑 (并行单元)
    """
    ticker_a, ticker_b, data_a, data_b = args
    try:
        # 协整检验
        _, pvalue, _ = coint(data_a, data_b)
        
        if pvalue < 0.05:
            # 计算回归残差和半衰期
            X = sm.add_constant(data_b)
            model = sm.OLS(data_a, X).fit()
            spread = model.resid
            
            # OU过程计算半衰期
            spread_lag = spread.shift(1)
            spread_diff = spread.diff()
            df_temp = pd.DataFrame({'diff': spread_diff, 'lag': spread_lag}).dropna()
            if not df_temp.empty:
                res_ou = sm.OLS(df_temp['diff'], df_temp['lag']).fit()
                theta = -res_ou.params.iloc[0]
                half_life = np.log(2) / theta if theta > 0 else 999
                
                if half_life < 60: # 过滤掉回归太慢的
                    return {
                        "pair": (ticker_a, ticker_b),
                        "p_value": pvalue,
                        "half_life": half_life,
                        "beta": model.params.iloc[1]
                    }
    except:
        pass
    return None

def run_full_sector_scan():
    # 定义扫描池 (核心板块龙头)
    computing = ["300308.SZ", "300394.SZ", "601138.SS", "002415.SZ", "300502.SZ"]
    semi = ["688012.SS", "600584.SS", "002156.SZ", "603986.SS", "688036.SS"]
    resources = ["601899.SS", "600547.SS", "600988.SS", "000960.SZ", "603993.SS"]
    
    all_tickers = computing + semi + resources
    print(f"📡 正在获取 {len(all_tickers)} 个标的的 3 年历史数据...")
    
    raw_data = yf.download(all_tickers, period="3y", progress=False)['Close']
    df = raw_data.ffill().dropna(axis=1, how='any')
    valid_tickers = df.columns.tolist()
    
    # 生成所有可能的两两组合
    combinations = list(itertools.combinations(valid_tickers, 2))
    print(f"🚀 准备并行扫描 {len(combinations)} 组配对...")
    
    # 准备并行参数
    task_args = []
    for t1, t2 in combinations:
        task_args.append((t1, t2, df[t1], df[t2]))
        
    start_time = time.time()
    cores = cpu_count()
    
    with Pool(processes=cores) as pool:
        results = pool.map(test_pair_cointegration, task_args)
        
    end_time = time.time()
    
    found_pairs = [r for r in results if r is not None]
    
    print(f"\n✅ 扫描完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"📊 发现协整配对总数: {len(found_pairs)}")
    
    # 输出报告
    print("\n" + "="*65)
    print(f"{'配对组合':<25} | {'P-Value':<10} | {'半衰期(d)':<10} | {'龙虾诊断'}")
    print("-" * 65)
    
    # 按 P 值排序展示前 10
    for r in sorted(found_pairs, key=lambda x: x['p_value'])[:10]:
        diag = "🔥 顶级" if r['p_value'] < 0.01 else "✅ 合格"
        p_str = f"{r['pair'][0]} vs {r['pair'][1]}"
        print(f"{p_str:<25} | {r['p_value']:.4f}     | {r['half_life']:.1f}       | {diag}")
    print("="*65)

if __name__ == "__main__":
    run_full_sector_scan()
