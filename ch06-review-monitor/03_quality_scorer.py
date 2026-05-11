"""
===============================================
第六章 示例 3：服务质量评分系统
===============================================
目标：建立多维度的服务质量评分体系，支持批量评估和趋势分析。

知识点：
    1. 服务质量评分模型（CSAT, FCR, AHT）
    2. 批量评估与趋势分析
    3. 可视化评分报告

运行方式：
    cd ch06-review-monitor
    python 03_quality_scorer.py
"""

import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# 服务质量指标
# ============================================================
class QualityMetrics(BaseModel):
    """单个会话的服务质量指标。"""
    session_id: str
    user_id: str

    # 核心指标
    csat_score: float = Field(ge=0, le=5, description="客户满意度 (CSAT) 1-5分")
    fcr: bool = Field(description="首次解决率 (FCR) - 是否一次解决")
    aht_seconds: float = Field(description="平均处理时间 (AHT) 秒")
    escalation_rate: bool = Field(description="是否转人工")

    # 辅助指标
    message_count: int = 0
    user_message_count: int = 0
    agent_message_count: int = 0
    has_ticket: bool = False
    has_escalation: bool = False

    # AI 特有指标
    rag_hit_count: int = 0          # RAG 命中次数
    rag_miss_count: int = 0         # RAG 未命中次数
    emotion_detected: str = ""      # 检测到的情绪
    routing_accuracy: bool = True   # 路由是否正确

    # 时间信息
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class QualityReport(BaseModel):
    """服务质量报告。"""
    report_period: str
    total_sessions: int = 0
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    # 核心指标汇总
    avg_csat: float = 0
    fcr_rate: float = 0            # 首次解决率
    avg_aht: float = 0             # 平均处理时间
    escalation_rate: float = 0     # 转人工率
    ticket_creation_rate: float = 0

    # AI 指标汇总
    avg_rag_accuracy: float = 0
    avg_routing_accuracy: float = 0

    # 趋势数据（按天）
    daily_csat: dict[str, float] = Field(default_factory=dict)
    daily_session_count: dict[str, int] = Field(default_factory=dict)

    # 改进建议
    improvement_suggestions: list[str] = Field(default_factory=list)


class QualityScorer:
    """
    服务质量评分系统。

    提供的指标：
    - CSAT (Customer Satisfaction Score): 客户满意度
    - FCR (First Contact Resolution): 首次解决率
    - AHT (Average Handling Time): 平均处理时间
    - Escalation Rate: 转人工率
    - RAG Accuracy: 知识库命中率
    """

    def __init__(self):
        self._metrics_history: list[QualityMetrics] = []

    def record_session(self, metrics: QualityMetrics):
        """记录一次会话的服务质量指标。"""
        self._metrics_history.append(metrics)

    def generate_report(self, period: str = "all") -> QualityReport:
        """生成服务质量报告。"""
        if not self._metrics_history:
            return QualityReport(report_period=period)

        metrics = self._metrics_history

        # 核心指标
        total = len(metrics)
        avg_csat = sum(m.csat_score for m in metrics) / total
        fcr_count = sum(1 for m in metrics if m.fcr)
        fcr_rate = fcr_count / total * 100
        avg_aht = sum(m.aht_seconds for m in metrics) / total
        escalation_count = sum(1 for m in metrics if m.has_escalation)
        escalation_rate = escalation_count / total * 100
        ticket_count = sum(1 for m in metrics if m.has_ticket)
        ticket_rate = ticket_count / total * 100

        # AI 指标
        rag_total = [m.rag_hit_count + m.rag_miss_count for m in metrics]
        rag_hits = [m.rag_hit_count for m in metrics]
        avg_rag_accuracy = (
            sum(rag_hits[i] / rag_total[i] * 100 if rag_total[i] > 0 else 0 for i in range(total))
            / total
        )
        routing_correct = sum(1 for m in metrics if m.routing_accuracy)
        avg_routing_accuracy = routing_correct / total * 100

        # 按天统计
        daily_csat = {}
        daily_count = {}
        for m in metrics:
            day = m.timestamp[:10]
            if day not in daily_csat:
                daily_csat[day] = []
                daily_count[day] = 0
            daily_csat[day].append(m.csat_score)
            daily_count[day] += 1

        # 改进建议
        suggestions = self._generate_suggestions(
            avg_csat, fcr_rate, escalation_rate, avg_rag_accuracy
        )

        return QualityReport(
            report_period=period,
            total_sessions=total,
            avg_csat=round(avg_csat, 2),
            fcr_rate=round(fcr_rate, 1),
            avg_aht=round(avg_aht, 1),
            escalation_rate=round(escalation_rate, 1),
            ticket_creation_rate=round(ticket_rate, 1),
            avg_rag_accuracy=round(avg_rag_accuracy, 1),
            avg_routing_accuracy=round(avg_routing_accuracy, 1),
            daily_csat={day: round(sum(scores) / len(scores), 2) for day, scores in daily_csat.items()},
            daily_session_count=daily_count,
            improvement_suggestions=suggestions,
        )

    def _generate_suggestions(self, csat, fcr, escalation, rag_accuracy) -> list[str]:
        """根据指标生成改进建议。"""
        suggestions = []
        if csat < 3.5:
            suggestions.append("CSAT 评分偏低，建议优化回答质量和共情能力")
        if fcr < 70:
            suggestions.append("首次解决率不足70%，建议优化知识库覆盖率和问题解决流程")
        if escalation > 20:
            suggestions.append("转人工率超过20%，建议分析转人工原因，增强 AI 处理能力")
        if rag_accuracy < 60:
            suggestions.append("知识库命中率偏低，建议补充知识库文档并优化 chunk 策略")
        if csat >= 4.0 and fcr >= 80:
            suggestions.append("整体表现优秀！持续关注用户反馈，定期更新知识库")
        if not suggestions:
            suggestions.append("各项指标正常，建议持续监控趋势变化")
        return suggestions

    def print_report(self, report: QualityReport):
        """打印可视化报告。"""
        print(f"\n{'═' * 50}")
        print(f"  📊 服务质量报告 - {report.report_period}")
        print(f"  生成时间: {report.generated_at[:19]}")
        print(f"{'═' * 50}")

        print(f"\n  总会话数: {report.total_sessions}")

        print(f"\n  ┌─ 核心指标 ─────────────────────────┐")
        print(f"  │ 客户满意度 (CSAT):    {'★' * int(report.avg_csat)}{'☆' * (5 - int(report.avg_csat))} {report.avg_csat}/5  │")
        print(f"  │ 首次解决率 (FCR):     {report.fcr_rate:5.1f}%             │")
        print(f"  │ 平均处理时间 (AHT):   {report.avg_aht:5.1f}秒            │")
        print(f"  │ 转人工率:             {report.escalation_rate:5.1f}%             │")
        print(f"  │ 工单创建率:           {report.ticket_creation_rate:5.1f}%             │")
        print(f"  └────────────────────────────────────┘")

        print(f"\n  ┌─ AI 指标 ───────────────────────────┐")
        print(f"  │ 知识库命中率:         {report.avg_rag_accuracy:5.1f}%             │")
        print(f"  │ 路由准确率:           {report.avg_routing_accuracy:5.1f}%             │")
        print(f"  └────────────────────────────────────┘")

        if report.daily_csat:
            print(f"\n  ┌─ 每日趋势 ─────────────────────────┐")
            for day, csat in sorted(report.daily_csat.items()):
                count = report.daily_session_count.get(day, 0)
                bar = "█" * int(csat * 4)
                print(f"  │ {day}  {bar} {csat} ({count}会话)  │")
            print(f"  └────────────────────────────────────┘")

        if report.improvement_suggestions:
            print(f"\n  💡 改进建议:")
            for i, suggestion in enumerate(report.improvement_suggestions, 1):
                print(f"     {i}. {suggestion}")


def demo():
    """演示服务质量评分系统。"""
    print("=" * 60)
    print("  第六章 示例 3：服务质量评分系统")
    print("=" * 60)

    scorer = QualityScorer()

    # 模拟 7 天的服务数据
    import random
    random.seed(42)

    base_date = datetime(2024, 1, 15)
    session_types = [
        # (名称, csat_base, fcr_prob, aht_base, esc_prob, rag_hit_prob)
        ("simple_inquiry", 4.2, 0.85, 60, 0.05, 0.9),
        ("product_issue", 3.5, 0.6, 120, 0.15, 0.7),
        ("complaint", 2.8, 0.3, 200, 0.4, 0.5),
        ("refund_request", 3.8, 0.7, 90, 0.1, 0.8),
        ("technical_help", 3.6, 0.5, 150, 0.2, 0.6),
    ]

    print("\n--- 模拟生成服务数据 ---\n")
    for day_offset in range(7):
        date = base_date + timedelta(days=day_offset)
        daily_sessions = random.randint(8, 15)

        for _ in range(daily_sessions):
            stype = random.choice(session_types)
            metrics = QualityMetrics(
                session_id=f"session-{date.strftime('%m%d')}-{random.randint(1000,9999)}",
                user_id=f"user-{random.randint(1,100)}",
                csat_score=max(1, min(5, stype[1] + random.uniform(-0.8, 0.8))),
                fcr=random.random() < stype[2],
                aht_seconds=stype[3] + random.uniform(-30, 30),
                escalation_rate=random.random() < stype[4],
                message_count=random.randint(4, 12),
                user_message_count=random.randint(2, 6),
                agent_message_count=random.randint(2, 6),
                has_ticket=random.random() < 0.3,
                has_escalation=random.random() < stype[4],
                rag_hit_count=random.randint(1, 3) if random.random() < stype[5] else 0,
                rag_miss_count=random.randint(0, 2),
                timestamp=date.isoformat(),
            )
            metrics.has_escalation = metrics.escalation_rate
            scorer.record_session(metrics)

        print(f"  {date.strftime('%Y-%m-%d')}: {daily_sessions} 个会话")

    # 生成报告
    print(f"\n--- 生成服务质量报告 ---")
    report = scorer.generate_report(period="2024-01-15 至 2024-01-21")
    scorer.print_report(report)

    print("\n✓ 第六章示例 3 运行完成！")
    print("  下一课：04_dashboard.py - 数据分析仪表盘")


if __name__ == "__main__":
    demo()
