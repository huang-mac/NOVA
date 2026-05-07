"""
===============================================
第四章 示例 2：意图分类模块
===============================================
目标：使用 LLM 对用户消息进行意图分类，决定走哪条处理路径。

知识点：
    1. 意图分类是客服路由的核心
    2. 使用 Pydantic 定义结构化输出
    3. 多标签分类（一条消息可能有多个意图）

运行方式：
    cd ch04-emotion-routing
    python 02_intent_classifier.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from pydantic import BaseModel, Field
from enum import Enum
from langchain_openai import ChatOpenAI
from config import get_llm_config


# ============================================================
# 意图分类数据模型
# ============================================================
class IntentType(str, Enum):
    """用户意图分类。"""
    # 产品咨询类
    PRODUCT_INQUIRY = "product_inquiry"     # 产品信息咨询
    PRICE_INQUIRY = "price_inquiry"         # 价格咨询

    # 售后服务类
    REFUND = "refund"                       # 退款
    RETURN = "return"                       # 退货
    EXCHANGE = "exchange"                   # 换货
    WARRANTY = "warranty"                   # 保修
    REPAIR = "repair"                       # 维修

    # 物流类
    DELIVERY = "delivery"                   # 配送/物流
    TRACKING = "tracking"                   # 订单/物流查询

    # 技术支持类
    TROUBLESHOOT = "troubleshoot"           # 故障排查
    USAGE_GUIDE = "usage_guide"             # 使用指导

    # 服务类
    COMPLAINT = "complaint"                 # 投诉
    ESCALATION = "escalation"               # 要求转人工
    FEEDBACK = "feedback"                   # 反馈/建议
    GREETING = "greeting"                   # 问候
    UNKNOWN = "unknown"                     # 无法识别


class IntentClassification(BaseModel):
    """意图分类结果。"""
    primary_intent: IntentType = Field(
        description="最主要的意图类别"
    )
    secondary_intent: list[IntentType] = Field(
        default_factory=list,
        description="次要意图列表（如有）"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分类置信度 0-1"
    )
    entities: dict[str, str] = Field(
        default_factory=dict,
        description="提取的关键实体，如产品名、订单号等"
    )
    summary: str = Field(
        description="一句话总结用户意图"
    )


class IntentClassifier:
    """
    用户意图分类器。

    将用户消息分类为预定义的意图类别，
    并提取关键实体信息。
    """

    # 意图对应的处理方式
    INTENT_ACTIONS = {
        IntentType.PRODUCT_INQUIRY: "rag_query",    # RAG 知识库检索
        IntentType.PRICE_INQUIRY: "rag_query",
        IntentType.REFUND: "create_ticket",          # 创建工单
        IntentType.RETURN: "create_ticket",
        IntentType.EXCHANGE: "create_ticket",
        IntentType.WARRANTY: "rag_query_or_ticket",  # 先查知识库，不行再建工单
        IntentType.REPAIR: "create_ticket",
        IntentType.DELIVERY: "rag_query",
        IntentType.TRACKING: "create_ticket",        # 查订单需要建工单
        IntentType.TROUBLESHOOT: "rag_query",
        IntentType.USAGE_GUIDE: "rag_query",
        IntentType.COMPLAINT: "escalate",            # 投诉直接转人工
        IntentType.ESCALATION: "escalate",           # 要求转人工
        IntentType.FEEDBACK: "create_ticket",        # 反馈记录为工单
        IntentType.GREETING: "greeting",             # 问候
        IntentType.UNKNOWN: "llm_fallback",          # 兜底由 LLM 回答
    }

    def __init__(self):
        config = get_llm_config()
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model_name"],
            temperature=0.1,
        )
        self.chain = self.llm.with_structured_output(IntentClassification)

    def classify(self, user_message: str) -> IntentClassification:
        """分类用户消息的意图。"""
        intent_descriptions = "\n".join(
            f"- {intent.value}: {intent.name}"
            for intent in IntentType
        )

        prompt = f"""分析用户消息的意图类别。

可选意图类别：
{intent_descriptions}

用户消息：「{user_message}」

请判断：
1. 主要意图是什么？
2. 有没有次要意图？
3. 置信度如何？
4. 消息中提到的实体（产品名、订单号、问题现象等）？
5. 一句话总结用户想做什么？
"""
        return self.chain.invoke(prompt)

    def get_action(self, intent: IntentType) -> str:
        """根据意图获取对应的处理动作。"""
        return self.INTENT_ACTIONS.get(intent, "llm_fallback")


def main():
    print("=" * 60)
    print("  第四章 示例 2：意图分类模块")
    print("=" * 60)

    classifier = IntentClassifier()

    # --------------------------------------------------
    # 测试不同意图的消息
    # --------------------------------------------------
    print("\n--- 意图分类测试 ---\n")

    test_messages = [
        "你好",
        "StarPods Pro 支持主动降噪吗？",
        "你们的耳机多少钱？",
        "我买的耳机有杂音，想退货",
        "订单 ORD-20240115-001 到哪了？",
        "你们的快递怎么这么慢！",
        "我要投诉你们的客服态度",
        "帮我转人工",
        "StarWatch X 怎么连接手机？",
        "建议你们增加夜间模式",
    ]

    for msg in test_messages:
        result = classifier.classify(msg)
        action = classifier.get_action(result.primary_intent)

        print(f"👤 用户: {msg}")
        print(f"   🎯 主意图: {result.primary_intent.value} ({result.primary_intent.name})")
        if result.secondary_intent:
            print(f"   🎯 次意图: {[i.value for i in result.secondary_intent]}")
        print(f"   📊 置信度: {result.confidence:.2f}")
        if result.entities:
            print(f"   🔑 实体: {result.entities}")
        print(f"   📝 总结: {result.summary}")
        print(f"   ⚙️ 路由到: {action}")
        print()

    # --------------------------------------------------
    # 路由决策总结
    # --------------------------------------------------
    print("--- 路由决策总结 ---")
    print("""
    意图 → 处理动作映射：

    ┌─────────────────┬──────────────────┬─────────────┐
    │     意图         │    处理动作        │   说明       │
    ├─────────────────┼──────────────────┼─────────────┤
    │ product_inquiry  │ rag_query        │ 知识库检索   │
    │ refund/return    │ create_ticket    │ 创建工单     │
    │ complaint        │ escalate         │ 转人工      │
    │ greeting         │ greeting         │ 问候回复     │
    │ unknown          │ llm_fallback     │ LLM 兜底    │
    └─────────────────┴──────────────────┴─────────────┘
    """)

    print("✓ 第四章示例 2 运行完成！")
    print("  下一课：03_smart_router.py - 智能路由系统")


if __name__ == "__main__":
    main()
