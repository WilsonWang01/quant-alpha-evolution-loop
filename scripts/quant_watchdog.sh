#!/bin/bash
# 🦞 龙虾炼丹炉看门狗 (Quant Evolution Watchdog)

LOG_FILE="/home/ubuntu/.openclaw/workspace/knowledge/quant_evolution_watchdog.log"
CRON_NAME="Quant Strategy Self-Evolution Engine"

# 检查最近一小时是否有炼丹日志更新 (检查通用的进化日志)
LAST_MOD=$(stat -c %Y /home/ubuntu/.openclaw/workspace/knowledge/quant_universal_evolution.md 2>/dev/null || echo 0)
CURRENT_TIME=$(date +%s)
DIFF=$((CURRENT_TIME - LAST_MOD))

echo "[$(date)] 检查炼丹炉 Loop 状态..." >> $LOG_FILE

if [ $DIFF -gt 4000 ]; then
    echo "⚠️ 炼丹炉 Loop 似乎灭火了（超过 66 分钟未更新）。正在尝试重启任务..." >> $LOG_FILE
    # 执行最新版本的演化脚本
    /usr/bin/python3 /home/ubuntu/.openclaw/workspace/scripts/quant_auto_evolution.py --break-system-packages >> $LOG_FILE 2>&1
    echo "✅ 唤醒指令已发送。" >> $LOG_FILE
else
    echo "🟢 炼丹炉火候正旺，运行正常。" >> $LOG_FILE
fi
