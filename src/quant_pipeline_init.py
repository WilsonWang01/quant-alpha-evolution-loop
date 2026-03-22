import os
import akshare as ak
import chromadb
from datetime import datetime
import time

class QuantDataPipeline:
    def __init__(self, db_path="data/quant_cache"):
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="market_data")

    def cache_daily_snapshot(self):
        """抓取 A 股实时快照并缓存到向量库，作为'分钟级'模拟的基础"""
        print(f"🦞 正在执行数据管道同步 (Retry Mode): {datetime.now()}")
        
        for attempt in range(3):
            try:
                # 抓取实时行情 (东财接口)
                df = ak.stock_zh_a_spot_em()
                
                # 仅取成交额前 50 的票入库做核心缓存演示
                df_top = df.sort_values(by="成交额", ascending=False).head(50)
                
                ids, metadatas, documents = [], [], []
                timestamp = int(time.time())
                
                for _, row in df_top.iterrows():
                    symbol = row['代码']
                    ids.append(f"{symbol}_{timestamp}")
                    metadatas.append({
                        "symbol": symbol,
                        "name": row['名称'],
                        "price": float(row['最新价']),
                        "pct_chg": float(row['涨跌幅'])
                    })
                    documents.append(f"{row['名称']} ({symbol}) 价格 {row['最新价']}")

                self.collection.add(ids=ids, metadatas=metadatas, documents=documents)
                print(f"✅ 成功同步 {len(ids)} 条核心个股行情至本地缓存库。")
                return True
            except Exception as e:
                print(f"⚠️ 第 {attempt+1} 次同步失败: {e}，正在重试...")
                time.sleep(2)
        return False

if __name__ == "__main__":
    pipeline = QuantDataPipeline()
    pipeline.cache_daily_snapshot()
