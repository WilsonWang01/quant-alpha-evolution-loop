import os
import requests
import json
import pandas as pd
import akshare as ak
from datetime import datetime

# 基础配置
GEMINI_MODEL = "google/gemini-3-flash-preview" # 映射为当前环境可用的别名

def get_market_data():
    """获取全市场情绪基础数据"""
    try:
        # 获取 A 股实时快照
        df = ak.stock_zh_a_spot_em()
        up_count = len(df[df['涨跌幅'] > 0])
        down_count = len(df[df['涨跌幅'] < 0])
        sentiment_score = (up_count / (up_count + down_count)) * 100 if (up_count + down_count) > 0 else 50
        
        # 获取主力流向
        flow_df = ak.stock_individual_fund_flow_rank(indicator="今日")
        net_inflow = flow_df['今日主力净流入-净额'].sum() / 100000000 # 亿元
        
        return {
            "up": up_count,
            "down": down_count,
            "sentiment": sentiment_score,
            "inflow": net_inflow,
            "top_sectors": flow_df.head(3)['名称'].tolist()
        }
    except Exception as e:
        print(f"数据抓取异常: {e}")
        return None

def llm_summarize(data):
    """调用 Gemini 模型进行情绪总结"""
    prompt = f"""
    作为数字龙虾(Digital Lobster)🦞，请根据以下 A 股市场数据，用幽默、简明、带点冷笑话的风格生成一段 100 字以内的情绪简报。
    
    数据：
    - 上涨家数: {data['up']}
    - 下跌家数: {data['down']}
    - 情绪指数: {data['sentiment']:.2f}/100
    - 主力净流入: {data['inflow']:.2f} 亿元
    - 活跃板块: {', '.join(data['top_sectors'])}
    
    要求：
    1. 风格必须是数字龙虾🦞（幽默、尖锐但有趣）。
    2. 不要废话，直接出结果。
    """
    
    # 模拟内部工具调用 (在 OpenClaw 中通常通过 sessions_spawn 运行，这里作为脚本演示逻辑)
    print(f"\n[LLM Prompt]: {prompt[:100]}...")
    # 实际部署时，这里会通过 OpenClaw 的 message 工具或 API 节点转发
    return "🦞 龙虾正在构思段子..."

if __name__ == "__main__":
    market_info = get_market_data()
    if market_info:
        summary = llm_summarize(market_info)
        print(summary)
