"""
===============================================
第四章 示例 3：智能路由系统
===============================================
目标：将情绪识别 + 意图分类 + 处理逻辑整合为完整的智能路由系统。

知识点：
    1. Router 模式：根据情绪和意图决定处理路径
    2. 多条件决策逻辑
    3. 优雅降级策略

运行方式：
    cd ch04-emotion-routing
    python 03_smart_router.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm_config

# 导入前两个示例中的类（实际项目中应放到独立模块中）
from typing import Any


# ============================================================
# 简化的数据模型（避免跨文件依赖）
# ============================================================
class EmotionType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    SAD = "sad"


class IntentType(str, Enum):
    PRODUCT_INQUIRY = "product_inquiry"
    PRICE_INQUIRY = "price_inquiry"
    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    WARRANTY = "warranty"
    REPAIR = "repair"
    DELIVERY = "delivery"
    TRACKING = "tracking"
    TROUBLESHOOT = "troubleshoot"
    USAGE_GUIDE = "usage_guide"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"
    GREETING = "greeting"
    UNKNOWN = "unknown"


class EmotionResult(BaseModel):
    emotion_type: EmotionType
    intensity: int = Field(ge=1, le=10)
    should_escalate: bool = False


class IntentResult(BaseModel):
    primary_intent: IntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: dict[str, str] = Field(default_factory=dict)
    summary: str = ""


# ============================================================
# 路由决策结果
# ============================================================
class RouteAction(str, Enum):
    """路由动作类型。"""
    GREETING = "greeting"              # 问候回复
    RAG_QUERY = "rag_query"            # 知识库检索回答
    CREATE_TICKET = "create_ticket"    # 创建工单
    ESCALATE = "escalate"              # 转人工
    LLM_FALLBACK = "llm_fallback"      # LLM 兜底回答


class RoutingDecision(BaseModel):
    """路由决策结果。"""
    action: RouteAction = Field(description="决定执行的动作")
    reason: str = Field(description="决策理由")
    priority: str = Field(default="normal", description="优先级: normal/high/urgent")
    emotion_prefix: str = Field(default="", description="情绪安抚前缀（如有）")


# ============================================================
# 智能路由器
# ============================================================
class SmartRouter:
    """
    智能路由器：综合情绪和意图做出路由决策。

    决策优先级（从高到低）：
    1. 紧急情绪（愤怒强度 >= 8）→ 直接转人工
    2. 明确要求转人工 → 转人工
    3. 投诉类意图 → 转人工
    4. 售后工单类（退款/退货/维修）→ 创建工单
    5. 咨询类（产品/价格/使用）→ 知识库检索
    6. 未知意图 → LLM 兜底
    """

    def __init__(self):
        config = get_llm_config()
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model_name"],
            temperature=0.3,
        )

        # 情绪安抚话术
        self.emotion_prefixes = {
            (EmotionType.ANGRY, "high"): "非常抱歉给您带来不好的体验，我理解您的心情。",
            (EmotionType.FRUSTRATED, "high"): "抱歉让您感到不满，我马上帮您处理。",
            (EmotionType.ANXIOUS, "high"): "理解您的焦急，我这边立即为您查询。",
            (EmotionType.SAD, "high"): "很抱歉给您带来了不好的体验，我们会尽力帮您解决。",
        }

    def route(self, emotion: EmotionResult, intent: IntentResult) -> RoutingDecision:
        """
        根据情绪和意图做出路由决策。

        Args:
            emotion: 情绪分析结果
            intent: 意图分类结果

        Returns:
            RoutingDecision: 路由决策结果
        """
        priority = "normal"
        reason = ""
        action = RouteAction.LLM_FALLBACK
        emotion_prefix = ""

        # --- 优先级 1：紧急情绪 → 转人工 ---
        if (emotion.emotion_type == EmotionType.ANGRY and emotion.intensity >= 8) \
                or emotion.should_escalate:
            return RoutingDecision(
                action=RouteAction.ESCALATE,
                reason=f"用户情绪强烈（{emotion.emotion_type.value}, "
                       f"强度{emotion.intensity}），建议转人工处理",
                priority="urgent",
                emotion_prefix="非常抱歉给您带来不好的体验！我立即为您安排人工客服跟进处理。",
            )

        # --- 优先级 2：明确要求转人工 ---
        if intent.primary_intent in (IntentType.ESCALATION, IntentType.COMPLAINT):
            return RoutingDecision(
                action=RouteAction.ESCALATE,
                reason=f"用户意图为 {intent.primary_intent.value}，需要人工处理",
                priority="high",
                emotion_prefix="好的，我马上为您转接人工客服，请稍等。",
            )

        # --- 优先级 3：售后工单类 → 创建工单 ---
        ticket_intents = {
            IntentType.REFUND, IntentType.RETURN, IntentType.EXCHANGE,
            IntentType.REPAIR, IntentType.TRACKING, IntentType.FEEDBACK,
        }
        if intent.primary_intent in ticket_intents:
            priority = "high" if emotion.intensity >= 6 else "normal"
            emotion_prefix = self._get_emotion_prefix(emotion, priority)
            return RoutingDecision(
                action=RouteAction.CREATE_TICKET,
                reason=f"用户意图为 {intent.primary_intent.value}（{intent.summary}），"
                       f"需要创建工单处理",
                priority=priority,
                emotion_prefix=emotion_prefix,
            )

        # --- 优先级 4：咨询类 → 知识库检索 ---
        rag_intents = {
            IntentType.PRODUCT_INQUIRY, IntentType.PRICE_INQUIRY,
            IntentType.DELIVERY, IntentType.TROUBLESHOOT,
            IntentType.USAGE_GUIDE, IntentType.WARRANTY,
        }
        if intent.primary_intent in rag_intents:
            priority = "high" if emotion.intensity >= 7 else "normal"
            emotion_prefix = self._get_emotion_prefix(emotion, priority)
            return RoutingDecision(
                action=RouteAction.RAG_QUERY,
                reason=f"用户意图为 {intent.primary_intent.value}（{intent.summary}），"
                       f"通过知识库检索回答",
                priority=priority,
                emotion_prefix=emotion_prefix,
            )

        # --- 优先级 5：问候 ---
        if intent.primary_intent == IntentType.GREETING:
            return RoutingDecision(
                action=RouteAction.GREETING,
                reason="用户发送问候",
                priority="normal",
            )

        # --- 优先级 6：兜底 ---
        return RoutingDecision(
            action=RouteAction.LLM_FALLBACK,
            reason=f"意图 {intent.primary_intent.value} 无明确路由规则，由 LLM 处理",
            priority="normal",
        )

    def _get_emotion_prefix(self, emotion: EmotionResult, priority: str) -> str:
        """根据情绪获取安抚话术。"""
        if priority == "high":
            key = (emotion.emotion_type, "high")
            return self.emotion_prefixes.get(key, "")
        return ""

    def format_emotion_prompt(self, user_message: str) -> str:
        """根据用户消息生成情绪分析提示词。"""
        return f"""分析以下用户消息的情绪（JSON格式）。
用户消息：「{user_message}」
返回: emotion_type(str), intensity(1-10), should_escalate(bool)"""


def demo():
    """演示智能路由系统的决策过程。"""
    print("=" * 60)
    print("  第四章 示例 3：智能路由系统")
    print("=" * 60)

    router = SmartRouter()

    # 模拟各种场景
    test_scenarios = [
        {
            "message": "你们好",
            "emotion": EmotionResult(emotion_type=EmotionType.NEUTRAL, intensity=2),
            "intent": IntentResult(primary_intent=IntentType.GREETING, confidence=0.95, summary="打招呼"),
        },
        {
            "message": "StarPods Pro 支持什么蓝牙版本？",
            "emotion": EmotionResult(emotion_type=EmotionType.NEUTRAL, intensity=2),
            "intent": IntentResult(primary_intent=IntentType.PRODUCT_INQUIRY, confidence=0.92, summary="咨询耳机蓝牙版本"),
        },
        {
            "message": "我买的耳机有杂音，要退货",
            "emotion": EmotionResult(emotion_type=EmotionType.FRUSTRATED, intensity=5),
            "intent": IntentResult(primary_intent=IntentType.RETURN, confidence=0.88, summary="耳机有杂音要求退货"),
        },
        {
            "message": "订单三天了还没发货！太慢了！！",
            "emotion": EmotionResult(emotion_type=EmotionType.ANGRY, intensity=8),
            "intent": IntentResult(primary_intent=IntentType.TRACKING, confidence=0.85, summary="催促订单发货"),
        },
        {
            "message": "你们这什么破客服，问了三遍了都没解决！我要投诉！",
            "emotion": EmotionResult(emotion_type=EmotionType.ANGRY, intensity=10, should_escalate=True),
            "intent": IntentResult(primary_intent=IntentType.COMPLAINT, confidence=0.95, summary="投诉客服未解决问题"),
        },
        {
            "message": "StarWatch X 怎么测血氧？",
            "emotion": EmotionResult(emotion_type=EmotionType.NEUTRAL, intensity=2),
            "intent": IntentResult(primary_intent=IntentType.USAGE_GUIDE, confidence=0.90, summary="询问手表血氧功能使用方法"),
        },
    ]

    for scenario in test_scenarios:
        msg = scenario["message"]
        emotion = scenario["emotion"]
        intent = scenario["intent"]

        decision = router.route(emotion, intent)

        # 动作 emoji 映射
        action_emoji = {
            RouteAction.GREETING: "👋",
            RouteAction.RAG_QUERY: "🔍",
            RouteAction.CREATE_TICKET: "📋",
            RouteAction.ESCALATE: "🚨",
            RouteAction.LLM_FALLBACK: "🤖",
        }

        print(f"👤 用户: {msg}")
        print(f"   情绪: {emotion.emotion_type.value}(强度{emotion.intensity}) | "
              f"意图: {intent.primary_intent.value}")
        print(f"   {action_emoji.get(decision.action, '')} 路由: {decision.action.value} | "
              f"优先级: {decision.priority}")
        print(f"   💡 理由: {decision.reason}")
        if decision.emotion_prefix:
            print(f"   💬 安抚: {decision.emotion_prefix}")
        print()

    # 路由流程图
    print("--- 路由决策流程 ---")
    print("""
    用户消息
      │
      ├──→ 情绪识别（EmotionDetector）
      │
      ├──→ 意图分类（IntentClassifier）
      │
      └──→ 智能路由（SmartRouter）
            │
            ├── 愤怒/投诉/要求转人工 ──→ 🚨 转人工（ESCALATE）
            │
            ├── 退款/退货/维修/查询 ──→ 📋 创建工单（CREATE_TICKET）
            │
            ├── 产品/价格/使用咨询 ──→ 🔍 知识库检索（RAG_QUERY）
            │
            ├── 问候 ────────────────→ 👋 问候回复（GREETING）
            │
            └── 其他 ────────────────→ 🤖 LLM兜底（LLM_FALLBACK）
    """)

    print("✓ 第四章示例 3 运行完成！")
    print("  下一章：ch05-ticket-escalation - 工单系统与转人工")


if __name__ == "__main__":
    demo()
