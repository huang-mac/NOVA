"""
===============================================
第六章 示例 4：数据分析仪表盘（终端版）
===============================================
目标：生成终端可视化的数据分析报告，展示客服系统的运行状态。

知识点：
    1. 数据聚合与统计
    2. 终端可视化图表（ASCII Art）
    3. 运营报告生成

运行方式：
    cd ch06-review-monitor
    python 04_dashboard.py
"""

import random
from datetime import datetime, timedelta
from collections import Counter


def generate_dashboard():
    """生成终端版数据分析仪表盘。"""

    # 模拟数据
    random.seed(42)

    # 会话统计
    total_sessions = 156
    avg_messages = 6.8
    avg_duration_min = 4.2

    # 情绪分布
    emotion_distribution = {
        "neutral": 68, "positive": 35, "confused": 18,
        "frustrated": 20, "angry": 10, "anxious": 5,
    }

    # 意图分布
    intent_distribution = {
        "product_inquiry": 38, "troubleshoot": 25, "return": 18,
        "delivery": 22, "complaint": 12, "refund": 15,
        "greeting": 14, "other": 12,
    }

    # 每小时对话量
    hourly_volume = {
        "09": 12, "10": 18, "11": 15, "12": 8, "13": 10,
        "14": 16, "15": 20, "16": 22, "17": 14, "18": 8,
    }

    # CSAT 趋势（最近7天）
    csat_trend = [3.8, 4.0, 3.9, 4.2, 4.1, 4.3, 4.2]
    days = ["01/15", "01/16", "01/17", "01/18", "01/19", "01/20", "01/21"]

    # ========================================================
    # 打印仪表盘
    # ========================================================
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           星辰科技 · 智能客服监控仪表盘                       ║")
    print("║           数据更新时间: 2024-01-21 18:00                      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                                                              ║")

    # --- 核心指标卡片 ---
    print("║  ┌─── 核心指标 ─────────────────────────────────────────┐   ║")
    print(f"║  │  总会话数      平均消息数    平均时长    工单创建率    │   ║")
    print(f"║  │  {total_sessions:>6}        {avg_messages:>5.1f}        {avg_duration_min:>4.1f} min     {28.2:>5.1f}%        │   ║")
    print(f"║  │  CSAT: {'█' * int(4.1*3)}{'░' * (15 - int(4.1*3))} 4.1/5.0                  │   ║")
    print(f"║  │  FCR:  {'█' * int(72*0.15)}{'░' * (15 - int(72*0.15))} 72.0%                     │   ║")
    print(f"║  │  转人工: {'█' * int(12*0.15)}{'░' * (15 - int(12*0.15))} 12.0%                     │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")

    print("║                                                              ║")

    # --- 情绪分布 ---
    print("║  ┌─── 用户情绪分布 ───────────────────────────────────────┐   ║")
    total_emotions = sum(emotion_distribution.values())
    for emotion, count in sorted(emotion_distribution.items(), key=lambda x: -x[1]):
        pct = count / total_emotions * 100
        bar_len = int(pct * 0.4)
        emoji = {"neutral": "😐", "positive": "😊", "confused": "🤔",
                 "frustrated": "😤", "angry": "😡", "anxious": "😰"}
        print(f"║  │  {emoji.get(emotion, '?')} {emotion:12s} {'█' * bar_len}{'░' * (20 - bar_len)} {pct:5.1f}% ({count:>3})   │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")

    print("║                                                              ║")

    # --- 意图分布 ---
    print("║  ┌─── 用户意图 Top 5 ────────────────────────────────────┐   ║")
    sorted_intents = sorted(intent_distribution.items(), key=lambda x: -x[1])[:5]
    for intent, count in sorted_intents:
        bar_len = int(count * 0.8)
        print(f"║  │  {intent:18s} {'█' * bar_len}{'░' * (20 - bar_len)} {count:>3}          │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")

    print("║                                                              ║")

    # --- CSAT 趋势 ---
    print("║  ┌─── CSAT 趋势（近7天）─────────────────────────────────┐   ║")
    max_csat = 5.0
    for i, (day, score) in enumerate(zip(days, csat_trend)):
        bar_len = int(score / max_csat * 15)
        trend = "↗" if i > 0 and score > csat_trend[i-1] else ("↘" if i > 0 and score < csat_trend[i-1] else "→")
        print(f"║  │  {day} {'█' * bar_len}{'░' * (15 - bar_len)} {score:.1f} {trend}      │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")

    print("║                                                              ║")

    # --- 每小时对话量 ---
    print("║  ┌─── 今日对话量（按时段）────────────────────────────────┐   ║")
    max_vol = max(hourly_volume.values())
    for hour, vol in hourly_volume.items():
        bar_len = int(vol / max_vol * 18)
        print(f"║  │  {hour}:00 {'█' * bar_len}{'░' * (18 - bar_len)} {vol:>3}           │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")

    print("║                                                              ║")

    # --- 告警信息 ---
    print("║  ┌─── 告警 ──────────────────────────────────────────────┐   ║")
    print("║  │  🟡 15:00-16:00 对话量峰值（22次），建议增加坐席       │   ║")
    print("║  │  🟠 CSAT 低于 3.5 的会话有 8 个，需人工复核             │   ║")
    print("║  │  🟢 知识库覆盖率 85.3%，状态良好                       │   ║")
    print("║  └───────────────────────────────────────────────────────┘   ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def main():
    print("=" * 60)
    print("  第六章 示例 4：数据分析仪表盘")
    print("=" * 60)

    generate_dashboard()

    print("✓ 第六章示例 4 运行完成！")
    print("  下一章：ch07-function-call - Function Call 与 Tool Use")


if __name__ == "__main__":
    main()
