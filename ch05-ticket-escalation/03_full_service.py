"""
===============================================
第五章 示例 3：完整售后流程集成
===============================================
目标：将工单系统、转人工流程与前面的情绪/意图识别整合为完整的售后处理流水线。

知识点：
    1. 完整的服务处理流水线
    2. 模块化设计：各组件松耦合
    3. 处理结果统一封装

运行方式：
    cd ch05-ticket-escalation
    python 03_full_service.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 导入前面章节的模块（简化版，避免跨文件依赖问题）
# ============================================================
# 实际项目中，应该通过 from xxx import xxx 导入
# 这里为了代码独立可运行，内联必要的类型定义

from enum import Enum


class RouteAction(str, Enum):
    GREETING = "greeting"
    RAG_QUERY = "rag_query"
    CREATE_TICKET = "create_ticket"
    ESCALATE = "escalate"
    LLM_FALLBACK = "llm_fallback"


class TicketCategory(str, Enum):
    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    REPAIR = "repair"
    DELIVERY = "delivery"
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    OTHER = "other"


class TicketPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# ============================================================
# 统一服务响应
# ============================================================
class ServiceResponse(BaseModel):
    """统一的服务响应格式。"""
    reply: str = Field(description="给用户的回复文本")
    action_taken: str = Field(description="执行的动作: greeting/rag/ticket/escalate/fallback")
    ticket_id: Optional[str] = Field(default=None, description="创建的工单ID（如有）")
    escalation_id: Optional[str] = Field(default=None, description="转人工请求ID（如有）")
    metadata: dict = Field(default_factory=dict, description="附加信息")


# ============================================================
# 简化的组件（模拟前几章的功能）
# ============================================================
class SimpleEmotionAnalyzer:
    """简化的情绪分析器（基于关键词，实际用 LLM 版本）。"""
    EMOTION_KEYWORDS = {
        "angry": ["垃圾", "破", "投诉", "骗子", "退款", "滚", "废物", "恶心"],
        "frustrated": ["慢", "等了", "还没", "多次", "又坏了", "什么破"],
        "anxious": ["急", "快点", "赶紧", "什么时候", "催"],
        "sad": ["失望", "难过", "伤心", "心寒"],
        "positive": ["好的", "谢谢", "满意", "不错", "好用"],
        "confused": ["怎么", "不知道", "看不懂", "不会"],
        "neutral": [],
    }

    def analyze(self, message: str) -> dict:
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            scores[emotion] = sum(1 for kw in keywords if kw in message)

        if not any(scores.values()):
            return {"emotion": "neutral", "intensity": 2, "should_escalate": False}

        top_emotion = max(scores, key=scores.get)
        intensity = min(10, scores[top_emotion] * 3 + 2)
        return {
            "emotion": top_emotion,
            "intensity": intensity,
            "should_escalate": top_emotion == "angry" and intensity >= 8,
        }


class SimpleIntentClassifier:
    """简化的意图分类器（基于关键词）。"""
    INTENT_KEYWORDS = {
        "product_inquiry": ["什么", "怎么样", "参数", "功能", "支持"],
        "price_inquiry": ["多少钱", "价格", "优惠", "折扣"],
        "refund": ["退款"],
        "return": ["退货"],
        "exchange": ["换货", "换新"],
        "warranty": ["保修", "质保"],
        "delivery": ["发货", "快递", "配送", "物流"],
        "tracking": ["到哪", "订单", "物流查询"],
        "complaint": ["投诉"],
        "escalation": ["转人工", "人工客服", "真人"],
        "greeting": ["你好", "在吗", "hello", "hi"],
    }

    def classify(self, message: str) -> dict:
        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in message for kw in keywords):
                return {"primary_intent": intent, "confidence": 0.8, "entities": {}}
        return {"primary_intent": "unknown", "confidence": 0.3, "entities": {}}


class SimpleRouter:
    """简化的智能路由器。"""
    def route(self, emotion: dict, intent: dict) -> dict:
        if emotion.get("should_escalate") or intent["primary_intent"] in ("complaint", "escalation"):
            return {"action": "escalate", "priority": "urgent" if emotion.get("should_escalate") else "high"}

        ticket_intents = {"refund", "return", "exchange", "repair", "tracking"}
        if intent["primary_intent"] in ticket_intents:
            pri = "high" if emotion["intensity"] >= 6 else "normal"
            return {"action": "create_ticket", "priority": pri}

        rag_intents = {"product_inquiry", "price_inquiry", "delivery", "warranty"}
        if intent["primary_intent"] in rag_intents:
            return {"action": "rag_query", "priority": "normal"}

        if intent["primary_intent"] == "greeting":
            return {"action": "greeting", "priority": "normal"}

        return {"action": "llm_fallback", "priority": "normal"}


# ============================================================
# 完整服务流水线
# ============================================================
class CustomerServicePipeline:
    """
    完整的客服服务流水线。

    处理流程：
    用户消息 → 情绪分析 → 意图分类 → 智能路由 → 执行动作 → 返回响应
    """

    def __init__(self):
        self.emotion_analyzer = SimpleEmotionAnalyzer()
        self.intent_classifier = SimpleIntentClassifier()
        self.router = SimpleRouter()

        # 模拟的组件
        self.ticket_counter = 0
        self.escalation_counter = 0

        # RAG 模拟知识库
        self.faq_answers = {
            "StarPods Pro": "StarPods Pro 是我们的旗舰降噪耳机，支持 48dB 主动降噪，蓝牙 5.3，30 小时续航，售价 ¥799。",
            "StarWatch X": "StarWatch X 是智能手表，支持心率血氧监测，7 天续航，IP68 防水，售价 ¥1299。",
            "退货": "7 天内可无理由退货，请在「我的订单」中申请或联系客服处理。",
            "退款": "退款将在 3-5 个工作日内原路到账。",
            "保修": "所有产品提供 1 年官方保修，非人为损坏可免费维修。",
            "发货": "默认顺丰包邮，1-3 个工作日送达。",
        }

    def process(self, user_message: str, session_id: str = "default",
                user_id: str = "anonymous") -> ServiceResponse:
        """
        处理用户消息的完整流水线。

        Args:
            user_message: 用户输入
            session_id: 会话ID
            user_id: 用户ID

        Returns:
            ServiceResponse: 统一的服务响应
        """
        print(f"\n  [Pipeline] 收到消息: {user_message}")

        # --- Step 1: 情绪分析 ---
        emotion = self.emotion_analyzer.analyze(user_message)
        print(f"  [Pipeline] 情绪分析: {emotion['emotion']} (强度{emotion['intensity']})")

        # --- Step 2: 意图分类 ---
        intent = self.intent_classifier.classify(user_message)
        print(f"  [Pipeline] 意图分类: {intent['primary_intent']}")

        # --- Step 3: 智能路由 ---
        routing = self.router.route(emotion, intent)
        action = routing["action"]
        priority = routing["priority"]
        print(f"  [Pipeline] 路由决策: {action} (优先级: {priority})")

        # --- Step 4: 执行动作 ---
        if action == "greeting":
            return self._handle_greeting(user_message)

        elif action == "rag_query":
            return self._handle_rag_query(user_message)

        elif action == "create_ticket":
            return self._handle_create_ticket(
                user_message, user_id, intent, emotion, priority
            )

        elif action == "escalate":
            return self._handle_escalation(
                user_message, session_id, user_id, intent, emotion, priority
            )

        else:
            return self._handle_fallback(user_message)

    def _handle_greeting(self, message: str) -> ServiceResponse:
        """处理问候。"""
        return ServiceResponse(
            reply="您好！我是星辰科技客服小星，请问有什么可以帮您？😊",
            action_taken="greeting",
        )

    def _handle_rag_query(self, message: str) -> ServiceResponse:
        """处理知识库查询。"""
        # 简单匹配（实际用向量检索）
        best_answer = "抱歉，我暂时没有找到相关信息，建议拨打售后热线 400-888-9999。"
        for keyword, answer in self.faq_answers.items():
            if keyword in message:
                best_answer = answer
                break

        return ServiceResponse(
            reply=best_answer,
            action_taken="rag_query",
            metadata={"query": message},
        )

    def _handle_create_ticket(self, message, user_id, intent, emotion, priority) -> ServiceResponse:
        """处理工单创建。"""
        self.ticket_counter += 1
        ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{self.ticket_counter:05d}"

        # 意图映射到工单分类
        category_map = {
            "refund": "refund", "return": "return", "exchange": "exchange",
            "repair": "repair", "tracking": "delivery",
        }
        category = category_map.get(intent["primary_intent"], "other")

        emotion_prefix = ""
        if emotion["intensity"] >= 6:
            emotion_prefix = "抱歉给您带来不好的体验，"

        reply = (
            f"{emotion_prefix}我已为您创建工单 {ticket_id}。"
            f"我们的客服团队会在 1-2 个工作日内与您联系。"
        )

        return ServiceResponse(
            reply=reply,
            action_taken="create_ticket",
            ticket_id=ticket_id,
            metadata={
                "category": category,
                "priority": priority,
                "emotion": emotion["emotion"],
            },
        )

    def _handle_escalation(self, message, session_id, user_id, intent, emotion, priority) -> ServiceResponse:
        """处理转人工。"""
        self.escalation_counter += 1
        escalation_id = f"ESC-{self.escalation_counter:04d}"

        reply = "非常抱歉给您带来不好的体验！我正在为您转接人工客服，请稍候..."
        if priority == "urgent":
            reply = "非常抱歉！我已为您加急转接人工客服，坐席会尽快为您处理，请稍候。"

        return ServiceResponse(
            reply=reply,
            action_taken="escalate",
            escalation_id=escalation_id,
            metadata={
                "reason": intent["primary_intent"],
                "priority": priority,
                "emotion": emotion["emotion"],
                "intensity": emotion["intensity"],
            },
        )

    def _handle_fallback(self, message: str) -> ServiceResponse:
        """兜底处理。"""
        return ServiceResponse(
            reply="感谢您的咨询！关于这个问题，我需要进一步核实。您可以拨打我们的售后热线 400-888-9999（工作日 9:00-18:00）获取更详细的信息。",
            action_taken="llm_fallback",
        )


def demo():
    """演示完整服务流水线。"""
    print("=" * 60)
    print("  第五章 示例 3：完整售后流程集成")
    print("=" * 60)

    pipeline = CustomerServicePipeline()

    # 模拟各种用户消息
    test_messages = [
        ("你好", "session-001", "user-alice"),
        ("StarPods Pro 支持降噪吗？", "session-001", "user-alice"),
        ("我买的耳机有杂音，要退货", "session-002", "user-bob"),
        ("你们什么破产品！刚买就坏了！我要投诉！", "session-003", "user-charlie"),
        ("帮我转人工", "session-004", "user-dave"),
        ("StarWatch X 多少钱？", "session-005", "user-eve"),
    ]

    print("\n" + "=" * 60)
    print("  开始处理用户消息")
    print("=" * 60)

    for message, session_id, user_id in test_messages:
        print(f"\n👤 用户 ({user_id}): {message}")
        response = pipeline.process(message, session_id, user_id)
        print(f"🤖 小星: {response.reply}")
        if response.ticket_id:
            print(f"  📋 工单号: {response.ticket_id} | 分类: {response.metadata.get('category')}")
        if response.escalation_id:
            print(f"  🚨 转人工ID: {response.escalation_id}")

    print(f"\n\n--- 处理统计 ---")
    print(f"  创建工单: {pipeline.ticket_counter} 个")
    print(f"  转人工: {pipeline.escalation_counter} 次")

    print("""
    完整处理流水线：

    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 用户消息  │───→│ 情绪分析  │───→│ 意图分类  │
    └──────────┘    └──────────┘    └────┬─────┘
                                       │
                                  ┌────▼─────┐
                                  │ 智能路由   │
                                  └────┬─────┘
                                       │
           ┌───────────┬───────────┬───┴───┬───────────┐
           ▼           ▼           ▼       ▼           ▼
        ┌──────┐  ┌──────┐  ┌──────┐ ┌──────┐  ┌──────┐
        │ 问候  │  │ RAG  │  │ 工单  │ │转人工│  │ 兜底  │
        └──┬───┘  └──┬───┘  └──┬───┘ └──┬───┘  └──┬───┘
           │         │         │        │         │
           └─────────┴─────────┴────────┴─────────┘
                              │
                         ┌────▼─────┐
                         │ 统一响应   │
                         └──────────┘
    """)

    print("\n✓ 第五章示例 3 运行完成！")
    print("  下一章：ch06-review-monitor - 复盘与质量监控")


if __name__ == "__main__":
    demo()
