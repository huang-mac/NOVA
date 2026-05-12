"""
===============================================
第一章 示例 3：基础客服对话机器人
===============================================
目标：实现一个能持续对话的客服机器人，具备以下能力：
    - 系统提示词定义客服角色
    - 循环对话（用户输入 → AI 回复）
    - 友好的欢迎语
    - 简单的意图识别（关键词匹配）
    - 退出指令

知识点：
    1. ChatPromptTemplate + LCEL 构建对话链
    2. while 循环实现持续对话
    3. 简单的关键词意图识别（后续章节会用 LLM 替代）

运行方式：
    cd ch01
    python 03_basic_chatbot.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import get_llm_config


# ============================================================
# 1. 客服系统提示词（核心 - 定义 AI 的全部行为）
# ============================================================
CUSTOMER_SERVICE_SYSTEM_PROMPT = """你是「星辰科技」的智能客服助手，名叫小星。

## 你的身份
- 名字：小星
- 公司：星辰科技（一家专注于智能硬件的公司）
- 产品线：智能耳机 StarPods Pro、智能手表 StarWatch X、智能音箱 StarBox

## 你的行为规范
1. 语气：专业但温暖，像一位经验丰富的客服
2. 回答：先确认用户问题，再给出解决方案
3. 长度：控制在 3 句话以内，除非用户追问细节
4. 格式：用编号列出步骤时使用 1. 2. 3.

## 你知道的信息
- 退换货政策：7天无理由，15天质量问题换新
- 保修政策：1年官方保修
- 配送时效：默认顺丰快递，1-3天送达
- 售后热线：400-888-9999
- 人工服务时间：工作日 9:00-18:00

## 你不知道的事（遇到时要说"这个问题我需要帮您转接人工客服"）
- 具体的订单状态查询
- 技术故障排查（硬件级别的）
- 价格谈判与折扣

## 特殊场景
- 用户表达不满时：先道歉，再提供解决方案
- 用户问"你是真人吗"：诚实回答"我是小星，星辰科技的智能客服助手"
- 用户说谢谢：简短回复"不客气，还有其他问题随时找我~"
"""


# ============================================================
# 2. 常见问题知识库（简单关键词匹配，后续用 RAG 替代）
# ============================================================
FAQ_KNOWLEDGE_BASE = {
    "退款": "我们支持7天无理由退款，退款将在3-5个工作日内原路到账。请问您的订单号是多少？我可以帮您查看。",
    "退货": "我们支持7天无理由退货，商品需保持完好。您可以在「我的订单」中申请退货，或告诉我订单号，我来帮您处理。",
    "换货": "15天内如遇质量问题可免费换新。请问您遇到了什么问题？请描述一下具体情况。",
    "保修": "我们的产品提供1年官方保修。如果您的产品出现非人为损坏的故障，可以联系我们进行免费维修。",
    "发货": "默认使用顺丰快递，下单后1-3个工作日送达。如需加急，可以选择次日达服务（需额外付费）。",
    "快递": "默认顺丰快递，1-3个工作日送达。下单后您会收到快递单号短信通知。",
    "价格": "我们的产品价格以官网和各大电商平台展示的价格为准，不定期有优惠活动，建议关注我们的官方账号。",
    "投诉": "非常抱歉给您带来不好的体验！请您详细描述遇到的问题，我会记录并安排专人跟进处理。",
}


# ============================================================
# 3. 简单意图识别函数（关键词匹配版）
# ============================================================
def simple_intent_detect(user_input: str) -> str | None:
    """
    通过关键词匹配识别用户意图。

    这是最简单的意图识别方式，后续章节会用 LLM + 结构化输出替代。
    实际生产中建议使用第四章的 LLM 意图分类方案。

    Args:
        user_input: 用户的输入文本

    Returns:
        str | None: 匹配到的意图关键词，未匹配返回 None
    """
    for keyword in FAQ_KNOWLEDGE_BASE:
        if keyword in user_input:
            return keyword
    return None


# ============================================================
# 4. 基础客服机器人类
# ============================================================
class BasicCustomerServiceBot:
    """
    基础客服对话机器人。

    这是本系列最简单的客服实现，后续章节会逐步增强：
    - 第二章：添加记忆系统，支持多轮对话
    - 第三章：添加 RAG 知识库检索
    - 第四章：添加 LLM 意图识别和情绪分析
    - 第五章：添加工单生成和转人工
    - 第六章：添加复盘和监控
    """

    def __init__(self):
        """初始化客服机器人。"""
        # 加载 LLM 配置
        config = get_llm_config()
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model_name"],
            temperature=0.5,
        )

        # 构建对话提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CUSTOMER_SERVICE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),
        ])

        # 构建 LCEL 链：prompt | llm
        self.chain = self.prompt | self.llm

        # 对话历史（目前仅保存在内存中，第二章会改进）
        self.chat_history = []

    def welcome(self) -> str:
        """生成欢迎语。"""
        return (
            "\n"
            "╔══════════════════════════════════════╗\n"
            "║     星辰科技 · 智能客服小星 🌟         ║\n"
            "╠══════════════════════════════════════╣\n"
            "║  您好！我是小星，很高兴为您服务~       ║\n"
            "║  我可以帮您：                          ║\n"
            "║  · 查询退换货政策                      ║\n"
            "║  · 了解产品信息                        ║\n"
            "║  · 查看配送时效                        ║\n"
            "║  · 解答常见问题                        ║\n"
            "║                                      ║\n"
            "║  输入 '退出' 或 'quit' 结束对话         ║\n"
            "╚══════════════════════════════════════╝\n"
        )

    def chat(self, user_input: str) -> str:
        """
        处理用户输入并生成回复。

        处理流程：
        1. 检查退出指令
        2. 关键词匹配常见问题（快速响应）
        3. 未匹配则调用 LLM 生成回复
        4. 将对话记录存入历史

        Args:
            user_input: 用户输入的文本

        Returns:
            str: AI 的回复文本
        """
        # --- 检查退出指令 ---
        if user_input.strip().lower() in ("退出", "quit", "exit", "再见"):
            return "感谢您使用星辰科技客服服务，祝您生活愉快！再见~ 🌟"

        # --- 关键词匹配快速响应 ---
        intent = simple_intent_detect(user_input)
        if intent:
            # 命中知识库，直接返回预设答案
            faq_answer = FAQ_KNOWLEDGE_BASE[intent]
            # 同时也调用 LLM 让回复更自然
            ai_response = self.chain.invoke({
                "chat_history": self.chat_history,
                "user_input": f"用户问了关于「{intent}」的问题。请参考以下标准答案来回复：\n{faq_answer}",
            }).content
        else:
            # 未命中知识库，由 LLM 自由回答
            ai_response = self.chain.invoke({
                "chat_history": self.chat_history,
                "user_input": user_input,
            }).content

        # --- 记录对话历史 ---
        from langchain_core.messages import HumanMessage, AIMessage
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=ai_response))

        return ai_response


# ============================================================
# 5. 主程序：交互式对话循环
# ============================================================
def main():
    """运行基础客服对话机器人。"""
    print("=" * 60)
    print("  第一章 示例 3：基础客服对话机器人")
    print("=" * 60)

    # 初始化机器人
    bot = BasicCustomerServiceBot()

    # 显示欢迎语
    print(bot.welcome())

    # 交互循环
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再见~")
            break

        if not user_input:
            continue

        # 获取 AI 回复
        ai_response = bot.chat(user_input)
        print(f"\n🤖 小星: {ai_response}")

        # 检查是否退出
        if "再见" in ai_response and "感谢" in ai_response:
            break

    # 显示对话统计
    print(f"\n--- 本次对话统计 ---")
    print(f"总对话轮数: {len(bot.chat_history) // 2}")
    print(f"对话记录数: {len(bot.chat_history)} 条消息")


if __name__ == "__main__":
    main()
