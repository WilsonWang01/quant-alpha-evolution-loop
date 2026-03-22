import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def prepare_ml_v2_data(symbol, period="10y"):
    """
    ML 数据准备 V2：
    增加波动率归一化和更多的交叉因子
    """
    df = yf.download(symbol, period=period)
    if df.empty: return None
    
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].squeeze()
    
    # 因子：动量、波动率、距离均线的偏离度
    df['Ret_10'] = close.pct_change(10)
    df['Ret_60'] = close.pct_change(60)
    df['Vol_10'] = close.pct_change().rolling(10).std()
    df['Dist_MA20'] = (close / close.rolling(20).mean()) - 1
    df['Dist_MA120'] = (close / close.rolling(120).mean()) - 1
    
    # 标签：未来 20 天是否上涨 > 5% (波段操作)
    df['Target'] = (close.shift(-20) / close - 1 > 0.05).astype(int)
    
    features = ['Ret_10', 'Ret_60', 'Vol_10', 'Dist_MA20', 'Dist_MA120']
    df = df.dropna()
    
    return df, features

def train_and_backtest_v2(symbol, name):
    print(f"### 🧪 ML-V2 波段实验 :: {name} ({symbol}) ###")
    
    data_tuple = prepare_ml_v2_data(symbol)
    if not data_tuple: return
    df, features = data_tuple
    
    split = int(len(df) * 0.7) # 增加回测样本量
    train_df = df.iloc[:split]
    test_df = df.iloc[split:].copy()
    
    # 训练
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03)
    model.fit(train_df[features], train_df['Target'])
    
    # 预测并生存信号
    test_df['Prob'] = model.predict_proba(test_df[features])[:, 1]
    # 高胜率阈值
    test_df['Signal'] = np.where(test_df['Prob'] > 0.60, 1.0, 0.0)
    
    # 绩效
    test_df['Market_Ret'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Ret'] = test_df['Market_Ret'] * test_df['Signal'].shift(1)
    
    cum_mkt = (1 + test_df['Market_Ret']).cumprod()
    cum_strat = (1 + test_df['Strategy_Ret']).cumprod()

    print(f"回测收益: {(cum_strat.iloc[-1]-1)*100:.2f}% (基准: {(cum_mkt.iloc[-1]-1)*100:.2f}%)")
    print(f"最大回撤: {(cum_strat / cum_strat.expanding().max() - 1).min()*100:.2f}%")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    targets = [
        ("300308.SZ", "中际旭创"),
        ("NVDA", "英伟达"),
        ("BTC-USD", "比特币")
    ]
    for code, name in targets:
        train_and_backtest_v2(code, name)
