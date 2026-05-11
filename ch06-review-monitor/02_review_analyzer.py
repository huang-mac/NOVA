"""
===============================================
第六章 示例 2：对话复盘分析
===============================================
目标：对历史对话进行自动复盘，识别问题模式、改进点。

知识点：
    1. 对话复盘的分析维度
    2. LLM 辅助的对话分析
    3. 问题归类与模式识别

运行方式：
    cd ch06-review-monitor
    python 02_review_analyzer.py
"""

import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ============================================================
# 复盘分析模型
# ============================================================
class IssueSeverity(str, Enum):
    INFO = "info"          # 信息（正常）
    WARNING = "warning"    # 警告（可优化）
    ERROR = "error"        # 错误（需改进）
    CRITICAL = "critical"  # 严重（必须修复）


class ReviewIssue(BaseModel):
    """复盘发现的问题。"""
    severity: IssueSeverity
    category: str          # 问题类别
    description: str       # 问题描述
    suggestion: str        # 改进建议
    message_index: int     # 出现在第几条消息


class ConversationReview(BaseModel):
    """单次对话的复盘结果。"""
    session_id: str
    overall_score: float = Field(ge=0, le=100, description="综合评分 0-100")
    response_time_score: float = Field(description="响应及时性评分")
    accuracy_score: float = Field(description="回答准确性评分")
    empathy_score: float = Field(description="共情能力评分")
    resolution_score: float = Field(description="问题解决评分")
    issues: list[ReviewIssue] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list, description="亮点")
    summary: str = Field(description="复盘总结")


class ReviewAnalyzer:
    """
    对话复盘分析器。

    分析维度：
    1. 响应及时性：机器人是否及时回复
    2. 回答准确性：回答是否基于知识库
    3. 共情能力：是否识别并回应用户情绪
    4. 问题解决：是否真正解决了用户问题
    5. 对话质量：是否有无效重复、跑题等
    """

    def __init__(self):
        pass

    def analyze_conversation(self, session: dict) -> ConversationReview:
        """
        分析一次对话会话。

        Args:
            session: 会话数据（包含 messages 列表）

        Returns:
            ConversationReview: 复盘结果
        """
        messages = session.get("messages", [])
        issues = []
        highlights = []

        # --- 分析 1：响应及时性 ---
        # 检查是否有长回复（模拟延迟）
        response_time_score = self._check_response_time(messages)

        # --- 分析 2：回答准确性 ---
        accuracy_score, accuracy_issues = self._check_accuracy(messages)
        issues.extend(accuracy_issues)

        # --- 分析 3：共情能力 ---
        empathy_score, empathy_issues, empathy_highlights = self._check_empathy(messages)
        issues.extend(empathy_issues)
        highlights.extend(empathy_highlights)

        # --- 分析 4：问题解决 ---
        resolution_score, resolution_issues = self._check_resolution(messages, session)
        issues.extend(resolution_issues)

        # --- 计算综合评分 ---
        overall_score = (
            response_time_score * 0.2 +
            accuracy_score * 0.3 +
            empathy_score * 0.2 +
            resolution_score * 0.3
        )

        # --- 生成总结 ---
        summary = self._generate_summary(session, overall_score, issues, highlights)

        return ConversationReview(
            session_id=session.get("session_id", "unknown"),
            overall_score=round(overall_score, 1),
            response_time_score=round(response_time_score, 1),
            accuracy_score=round(accuracy_score, 1),
            empathy_score=round(empathy_score, 1),
            resolution_score=round(resolution_score, 1),
            issues=issues,
            highlights=highlights,
            summary=summary,
        )

    def _check_response_time(self, messages: list) -> float:
        """检查响应及时性。"""
        # 这里简化处理，实际应分析时间戳差值
        return 85.0  # 模拟得分

    def _check_accuracy(self, messages: list) -> tuple[float, list]:
        """检查回答准确性。"""
        issues = []
        score = 90.0

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if msg.get("role") == "agent":
                # 检测模糊回答
                vague_phrases = ["我不确定", "可能", "也许", "不太清楚"]
                for phrase in vague_phrases:
                    if phrase in content:
                        issues.append(ReviewIssue(
                            severity=IssueSeverity.WARNING,
                            category="准确性",
                            description=f"机器人使用了模糊表述「{phrase}」",
                            suggestion="应为模糊表述提供确定性回答或转人工",
                            message_index=i,
                        ))
                        score -= 5

                # 检测是否有信息编造嫌疑
                if "我查了一下" in content and "无法确认" not in content:
                    # 模拟：在实际中应与知识库交叉验证
                    pass

        return max(0, score), issues

    def _check_empathy(self, messages: list) -> tuple[float, list, list]:
        """检查共情能力。"""
        issues = []
        highlights = []
        score = 75.0

        # 检测用户负面情绪后机器人的回应
        negative_keywords = ["不满", "生气", "差", "慢", "投诉", "破", "垃圾", "退款"]
        empathy_keywords = ["抱歉", "理解", "抱歉给您", "不舒服", "体验不好", "改进"]

        for i in range(len(messages)):
            msg = messages[i]
            if msg.get("role") == "user":
                # 检测用户负面情绪
                has_negative = any(kw in msg.get("content", "") for kw in negative_keywords)
                if has_negative and i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "agent":
                        has_empathy = any(
                            kw in next_msg.get("content", "") for kw in empathy_keywords
                        )
                        if has_empathy:
                            highlights.append(
                                f"第{i}轮：用户表达不满后，机器人表达了共情"
                            )
                            score += 5
                        else:
                            issues.append(ReviewIssue(
                                severity=IssueSeverity.WARNING,
                                category="共情能力",
                                description=f"用户表达不满后，机器人未表达共情",
                                suggestion="应先道歉/共情，再处理问题",
                                message_index=i + 1,
                            ))
                            score -= 10

        return min(100, max(0, score)), issues, highlights

    def _check_resolution(self, messages: list, session: dict) -> tuple[float, list]:
        """检查问题是否解决。"""
        issues = []
        score = 80.0

        status = session.get("status", "")
        if status == "escalated":
            issues.append(ReviewIssue(
                severity=IssueSeverity.INFO,
                category="问题解决",
                description="会话已转人工，AI 未能独立解决",
                suggestion="分析转人工原因，优化 AI 处理能力",
                message_index=len(messages),
            ))
            score = 50.0
        elif status == "closed":
            # 检查是否创建了工单
            if session.get("ticket_id"):
                score = 85.0
            else:
                score = 90.0
                highlights = ["问题在 AI 对话中直接解决"]
        else:
            issues.append(ReviewIssue(
                severity=IssueSeverity.WARNING,
                category="问题解决",
                description="会话未正常关闭",
                suggestion="确保每次对话都有明确的结束状态",
                message_index=len(messages),
            ))

        return score, issues

    def _generate_summary(self, session, score, issues, highlights) -> str:
        """生成复盘总结。"""
        user_name = session.get("user_name", "未知用户")
        msg_count = session.get("total_messages", 0)

        summary_parts = [
            f"会话 {session.get('session_id', '')} 复盘：",
            f"用户 {user_name}，共 {msg_count} 条消息。",
            f"综合评分 {score} 分。",
        ]

        if highlights:
            summary_parts.append("亮点：" + "；".join(highlights))

        critical_issues = [i for i in issues if i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)]
        if critical_issues:
            summary_parts.append(f"需改进 {len(critical_issues)} 项：")
            for issue in critical_issues[:3]:
                summary_parts.append(f"  - [{issue.category}] {issue.description}")

        return "\n".join(summary_parts)


def demo():
    """演示对话复盘分析。"""
    print("=" * 60)
    print("  第六章 示例 2：对话复盘分析")
    print("=" * 60)

    analyzer = ReviewAnalyzer()

    # 模拟三个会话数据
    sessions = [
        {
            "session_id": "session-001",
            "user_id": "user-alice",
            "user_name": "Alice",
            "status": "closed",
            "total_messages": 6,
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "agent", "content": "您好！我是客服小星，请问有什么可以帮您？"},
                {"role": "user", "content": "StarPods Pro 多少钱？"},
                {"role": "agent", "content": "StarPods Pro 官方售价 ¥899，目前活动价 ¥799。"},
                {"role": "user", "content": "好的谢谢"},
                {"role": "agent", "content": "不客气，还有其他问题随时找我~"},
            ],
        },
        {
            "session_id": "session-002",
            "user_id": "user-bob",
            "user_name": "Bob",
            "status": "closed",
            "total_messages": 6,
            "ticket_id": "TK-20240120-00001",
            "messages": [
                {"role": "user", "content": "我的耳机有杂音！"},
                {"role": "agent", "content": "抱歉给您带来不便，请问是什么型号？"},
                {"role": "user", "content": "StarPods Pro"},
                {"role": "agent", "content": "抱歉给您带来不好的体验，我已为您创建退货工单。"},
                {"role": "user", "content": "好的"},
                {"role": "agent", "content": "工单已创建，客服会在1-2天联系您。"},
            ],
        },
        {
            "session_id": "session-003",
            "user_id": "user-charlie",
            "user_name": "Charlie",
            "status": "escalated",
            "total_messages": 4,
            "messages": [
                {"role": "user", "content": "你们什么破客服！问了三遍了！我要投诉！"},
                {"role": "agent", "content": "请问您想咨询什么问题？"},
                {"role": "user", "content": "退款！说了多少次了！"},
                {"role": "agent", "content": "正在为您转接人工客服..."},
            ],
        },
    ]

    # 逐个分析
    for session in sessions:
        review = analyzer.analyze_conversation(session)

        print(f"\n{'─' * 50}")
        print(f"📋 会话: {review.session_id} | 用户: {session['user_name']}")
        print(f"{'─' * 50}")
        print(f"  综合评分: {review.overall_score}")
        print(f"  响应及时性: {review.response_time_score} | "
              f"准确性: {review.accuracy_score} | "
              f"共情能力: {review.empathy_score} | "
              f"问题解决: {review.resolution_score}")

        if review.issues:
            print(f"\n  ⚠️ 发现 {len(review.issues)} 个问题:")
            for issue in review.issues:
                emoji = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}
                print(f"    {emoji.get(issue.severity.value, '⚪')} [{issue.severity.value}] "
                      f"{issue.category}: {issue.description}")
                print(f"       建议: {issue.suggestion}")

        if review.highlights:
            print(f"\n  ✨ 亮点:")
            for h in review.highlights:
                print(f"    · {h}")

        print(f"\n  📝 总结:")
        for line in review.summary.split("\n"):
            print(f"    {line}")

    print("\n✓ 第六章示例 2 运行完成！")
    print("  下一课：03_quality_scorer.py - 服务质量评分系统")


if __name__ == "__main__":
    demo()
