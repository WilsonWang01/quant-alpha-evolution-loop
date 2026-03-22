import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

def prepare_ml_data(symbol, period="5y"):
    """
    ML 数据准备：
    集成 WorldQuant Alphas 思路 + 常用技术指标
    目标：预测未来 5 天是否上涨超过 3%
    """
    df = yf.download(symbol, period=period)
    if df.empty: return None
    
    # 强制单级索引
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # --- 特征工程 (Alpha Factors) ---
    # 1. 动量因子 (Returns)
    df['Ret_5'] = close.pct_change(5)
    df['Ret_20'] = close.pct_change(20)
    
    # 2. 波动率因子
    df['Vol_20'] = close.pct_change().rolling(20).std()
    
    # 3. 量价因子 (Alpha 101 思路)
    # Alpha#6: -1 * correlation(open, volume, 10)
    df['Alpha6'] = -1 * df['Open'].rolling(10).corr(df['Volume'])
    
    # 4. 技术指标
    df['MA5_20_Gap'] = (close.rolling(5).mean() / close.rolling(20).mean()) - 1
    df['RSI'] = 100 - (100 / (1 + (close.diff().where(close.diff() > 0, 0).rolling(14).mean() / 
                                  close.diff().where(close.diff() < 0, 0).abs().rolling(14).mean())))

    # --- 标签生成 ---
    # 预测目标：未来 5 天的最大涨幅是否 > 2%
    df['Target'] = (close.shift(-5) / close - 1 > 0.02).astype(int)
    
    # 清洗数据
    features = ['Ret_5', 'Ret_20', 'Vol_20', 'Alpha6', 'MA5_20_Gap', 'RSI']
    df = df.dropna()
    
    return df, features

def train_and_backtest_ml(symbol, name):
    print(f"### 🧪 ML 策略实验报告 :: {name} ({symbol}) ###")
    
    data_tuple = prepare_ml_data(symbol)
    if not data_tuple: return
    df, features = data_tuple
    
    # 划分训练集和测试集 (前 80% 训练，后 20% 测试)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:].copy()
    
    X_train = train_df[features]
    y_train = train_df['Target']
    X_test = test_df[features]
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 模型训练 (XGBoost)
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 预测胜率 (概率)
    test_df['Prob'] = model.predict_proba(X_test_scaled)[:, 1]
    
    # --- ML 回测逻辑 ---
    # 策略：概率 > 0.55 时买入，否则空仓
    test_df['Signal'] = np.where(test_df['Prob'] > 0.55, 1.0, 0.0)
    
    test_df['Market_Ret'] = test_df['Close'].pct_change().fillna(0)
    test_df['Strategy_Ret'] = test_df['Market_Ret'] * test_df['Signal'].shift(1)
    
    cum_mkt = (1 + test_df['Market_Ret']).cumprod()
    cum_strat = (1 + test_df['Strategy_Ret'].fillna(0)).cumprod()
    
    total_mkt = (cum_mkt.iloc[-1] - 1) * 100
    total_strat = (cum_strat.iloc[-1] - 1) * 100
    alpha = total_strat - total_mkt
    
    mdd = (cum_strat / cum_strat.expanding().max() - 1).min() * 100

    print(f"训练周期: {train_df.index[0].date()} 至 {train_df.index[-1].date()}")
    print(f"回测周期: {test_df.index[0].date()} 至 {test_df.index[-1].date()}")
    print("-" * 50)
    print(f"基准收益: {total_mkt:.2f}%")
    print(f"ML 策略收益: {total_strat:.2f}%")
    print(f"超额 (Alpha): {alpha:+.2f}% {'🔥' if alpha > 0 else '❄️'}")
    print(f"策略最大回撤: {mdd:.2f}%")
    print("-" * 50)
    
    # 打印最重要的特征
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("📈 模型核心因子权重:")
    print(importances.to_string())
    print("=" * 50 + "\n")

if __name__ == "__main__":
    targets = [
        ("300308.SZ", "中际旭创"),
        ("NVDA", "英伟达"),
        ("001309.SZ", "德明利")
    ]
    for code, name in targets:
        try:
            train_and_backtest_ml(code, name)
        except Exception as e:
            print(f"❌ {name} 实验失败: {e}")
