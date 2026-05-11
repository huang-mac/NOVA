"""
===============================================
第四章 示例 1：情绪识别模块
===============================================
目标：使用 LLM 识别用户消息中的情绪状态，为后续智能路由提供依据。

知识点：
    1. 使用结构化输出（Pydantic）让 LLM 返回 JSON 格式的情绪分析结果
    2. with_structured_output() 方法约束 LLM 输出格式
    3. 情绪维度设计：情绪类别 + 强度 + 关键词

运行方式：
    cd ch04-emotion-routing
    python 01_emotion_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm_config


# ============================================================
# 情绪数据模型定义
# ============================================================
class EmotionType(str, Enum):
    """情绪类别枚举。"""
    POSITIVE = "positive"       # 积极/满意
    NEUTRAL = "neutral"         # 中性/平静
    CONFUSED = "confused"       # 困惑/不解
    FRUSTRATED = "frustrated"   # 烦躁/不满
    ANGRY = "angry"             # 愤怒/生气
    ANXIOUS = "anxious"         # 焦急/催促
    SAD = "sad"                 # 失望/难过


class EmotionAnalysis(BaseModel):
    """
    情绪分析结果的数据模型。

    使用 Pydantic 定义结构化输出格式，
    LangChain 的 with_structured_output() 会自动将 LLM 输出解析为此模型。
    """
    emotion_type: EmotionType = Field(
        description="用户的主要情绪类别"
    )
    intensity: int = Field(
        ge=1, le=10,
        description="情绪强度，1=轻微，10=极强"
    )
    keywords: list[str] = Field(
        description="反映情绪的关键词列表"
    )
    reason: str = Field(
        description="一句话说明判断依据"
    )
    should_escalate: bool = Field(
        description="是否建议转人工（愤怒/极度不满时建议转人工）"
    )


class EmotionDetector:
    """
    用户情绪检测器。

    使用 PydanticOutputParser 让 LLM 输出结构化 JSON，
    不依赖原生 response_format API，兼容更多模型/API。
    """

    def __init__(self):
        config = get_llm_config()
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model_name"],
            temperature=0.1,
        )

        self.parser = PydanticOutputParser(pydantic_object=EmotionAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("human", """分析以下用户消息的情绪状态。

用户消息：「{user_message}」

请判断：
1. 用户的主要情绪是什么？
2. 情绪强度如何？（1-10）
3. 哪些词语反映了这种情绪？
4. 判断依据是什么？
5. 是否需要转人工处理？（情绪强度>=8 或 情绪为 angry 时建议转人工）

{format_instructions}"""),
        ]).partial(
            format_instructions=self.parser.get_format_instructions(),
        )

        self.chain = self.prompt | self.llm | self.parser

    def analyze(self, user_message: str) -> EmotionAnalysis:
        """
        分析用户消息的情绪。
        """
        return self.chain.invoke({"user_message": user_message})

    def get_emotion_emoji(self, emotion_type: EmotionType) -> str:
        """根据情绪类型返回对应的 emoji。"""
        emoji_map = {
            EmotionType.POSITIVE: "😊",
            EmotionType.NEUTRAL: "😐",
            EmotionType.CONFUSED: "🤔",
            EmotionType.FRUSTRATED: "😤",
            EmotionType.ANGRY: "😡",
            EmotionType.ANXIOUS: "😰",
            EmotionType.SAD: "😢",
        }
        return emoji_map.get(emotion_type, "❓")


def main():
    print("=" * 60)
    print("  第四章 示例 1：情绪识别模块")
    print("=" * 60)

    detector = EmotionDetector()

    # --------------------------------------------------
    # 测试不同情绪的用户消息
    # --------------------------------------------------
    print("\n--- 情绪识别测试 ---\n")

    test_messages = [
        "你们的产品真的很好用，非常满意！",
        "请问退货的流程是什么？",
        "这个功能怎么用啊，看不懂说明...",
        "等了三天了还没发货，到底什么时候能发？！",
        "什么破产品！刚买就坏了，退款！！",
        "我的快递显示已签收但我没收到，急死了",
        "用了两个月就出问题了，很失望",
        "你们客服能不能专业一点？每次回答都不一样",
    ]

    for msg in test_messages:
        result = detector.analyze(msg)
        emoji = detector.get_emotion_emoji(result.emotion_type)

        print(f"👤 用户: {msg}")
        print(f"   {emoji} 情绪: {result.emotion_type.value} | "
              f"强度: {result.intensity}/10 | "
              f"转人工: {'是' if result.should_escalate else '否'}")
        print(f"   📝 关键词: {', '.join(result.keywords)}")
        print(f"   💡 依据: {result.reason}")
        print()

    # --------------------------------------------------
    # 情绪处理策略建议
    # --------------------------------------------------
    print("--- 情绪处理策略 ---")
    print("""
    ┌───────────────┬──────────┬──────────────────────────┐
    │    情绪类型    │  强度范围  │      处理策略              │
    ├───────────────┼──────────┼──────────────────────────┤
    │ positive      │  1-10    │ 表达感谢，推荐其他产品       │
    │ neutral       │  1-10    │ 正常回答问题               │
    │ confused      │  1-5     │ 详细解释，提供图文教程        │
    │ confused      │  6-10    │ 简化回答，主动询问哪步不懂    │
    │ frustrated    │  1-5     │ 正常处理 + 适当安抚          │
    │ frustrated    │  6-10    │ 优先处理 + 道歉 + 跟进承诺    │
    │ angry         │  1-7     │ 道歉 + 立即转人工            │
    │ angry         │  8-10    │ 道歉 + 立即转人工 + 升级处理  │
    │ anxious       │  1-10    │ 快速响应 + 提供进度查询方式   │
    │ sad           │  1-10    │ 共情 + 积极提供解决方案       │
    └───────────────┴──────────┴──────────────────────────┘
    """)

    print("✓ 第四章示例 1 运行完成！")
    print("  下一课：02_intent_classifier.py - 意图分类模块")


if __name__ == "__main__":
    main()
