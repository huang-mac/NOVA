"""
===============================================
第二章 示例 2：摘要记忆 (SummaryMemory)
===============================================
目标：手动实现 AI 自动总结对话要点，解决长对话 Token 溢出问题。

知识点：
    1. 用 LLM 定期将对话历史压缩为摘要
    2. 新对话时将摘要注入上下文，而非完整历史
    3. 适合需要长时间对话但 Token 有限的场景

注意：LangChain 1.x 已移除 ConversationSummaryMemory，
      这里用 LLM 手动实现摘要逻辑。

运行方式：
    cd ch02-memory
    python 02_summary_memory.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import get_llm_config


# 用于生成摘要的提示词
SUMMARY_PROMPT = """请用简洁的语言总结以下对话的要点（不超过200字），保留关键信息如：
- 用户身份（姓名、订单号等）
- 核心问题/诉求
- 已给出的建议/方案
- 待解决的问题

对话内容：
{conversation}

摘要："""


class SummaryMemory:
    """手动实现的摘要记忆，等价于已废弃的 ConversationSummaryMemory。"""

    def __init__(self, llm, max_history_before_summary=4):
        """
        Args:
            llm: 用于生成摘要的语言模型
            max_history_before_summary: 累积多少轮对话后触发一次摘要
        """
        self.llm = llm
        self.max_history_before_summary = max_history_before_summary
        self.summary = ""  # 当前摘要
        self.recent_messages = []  # 摘要之后的新消息
        self._turn_count = 0

    def save_context(self, user_input: str, ai_output: str):
        """保存一轮对话，必要时触发摘要。"""
        self.recent_messages.append(HumanMessage(content=user_input))
        self.recent_messages.append(AIMessage(content=ai_output))
        self._turn_count += 1

        if self._turn_count >= self.max_history_before_summary:
            self._summarize()

    def _summarize(self):
        """将当前所有消息压缩为摘要。"""
        # 构建要摘要的内容：旧摘要 + 新消息
        parts = []
        if self.summary:
            parts.append(f"之前的摘要：{self.summary}")
        recent_text = "\n".join(
            [f"{'用户' if m.type == 'human' else '客服'}: {m.content}"
             for m in self.recent_messages]
        )
        parts.append(f"最新对话：\n{recent_text}")
        conversation = "\n\n".join(parts)

        response = self.llm.invoke(
            SUMMARY_PROMPT.format(conversation=conversation)
        )
        self.summary = response.content
        self.recent_messages = []
        self._turn_count = 0

    def load_memory_variables(self):
        """返回当前记忆（摘要 + 最近未摘要的消息）。"""
        result = []
        if self.summary:
            result.append(SystemMessage(
                content=f"[对话历史摘要] {self.summary}"
            ))
        result.extend(self.recent_messages)
        return {"chat_history": result}


def main():
    print("=" * 60)
    print("  第二章 示例 2：摘要记忆 (SummaryMemory)")
    print("=" * 60)

    config = get_llm_config()

    # 对话 LLM
    chat_llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.5,
    )
    # 摘要 LLM（用更低温度保证稳定性）
    summary_llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.3,
    )

    # --------------------------------------------------
    # 创建摘要记忆（每 3 轮触发一次摘要）
    # --------------------------------------------------
    memory = SummaryMemory(llm=summary_llm, max_history_before_summary=3)

    # --------------------------------------------------
    # 模拟一段较长的客服对话
    # --------------------------------------------------
    print("\n--- 模拟长对话（观察摘要如何演变）---\n")

    long_conversation = [
        ("你好，我是张三", "张三您好！我是星辰科技客服小星，请问有什么可以帮您？"),
        ("我想咨询一下你们的 StarPods Pro 耳机",
         "StarPods Pro 是我们的旗舰降噪耳机，支持ANC主动降噪、30小时续航、蓝牙5.3。请问您想了解哪方面？"),
        ("价格是多少？", "StarPods Pro 官方售价 ¥899，目前新春活动价 ¥799，还赠送耳机保护套。"),
        ("续航怎么样？", "开启降噪约30小时，关闭降噪可达40小时。快充10分钟可听3小时。"),
        ("防水吗？", "支持IPX4防水等级，日常出汗和小雨都没问题，但不建议游泳时使用。"),
        ("我上周在你们天猫店买了一副",
         "好的，请问您的订单号是多少？我可以帮您查看订单详情。"),
        ("订单号是 ORD-20240110-888",
         "已查到您的订单，StarPods Pro 星空灰，1月10日下单，1月12日已签收。请问有什么问题吗？"),
        ("右耳有时候会断连",
         "这可能是蓝牙连接问题。建议您：1. 先删除设备配对记录 2. 重置耳机 3. 重新配对。如果问题持续，我们提供1年保修，可以安排换新。"),
    ]

    for i, (user_msg, ai_msg) in enumerate(long_conversation, 1):
        print(f"[第 {i} 轮]")
        print(f"  👤 {user_msg}")
        print(f"  🤖 {ai_msg}")

        memory.save_context(user_msg, ai_msg)

        # 每 3 轮显示一次当前摘要
        if i % 3 == 0 and memory.summary:
            print(f"\n  📋 [第 {i} 轮后 - 当前对话摘要]")
            print(f"  {memory.summary}")
            print()

    # --------------------------------------------------
    # 对比 BufferMemory vs SummaryMemory 的 Token 消耗
    # --------------------------------------------------
    print("\n--- 对比：完整记忆 vs 摘要记忆 ---\n")

    # 摘要记忆的总字符数
    memory_vars = memory.load_memory_variables()
    summary_text = "\n".join(
        [msg.content for msg in memory_vars["chat_history"]]
    )
    print(f"摘要记忆总字符数: {len(summary_text)} 字符")

    # 完整记忆的字符数
    buffer_text = "\n".join([f"{u}\n{a}" for u, a in long_conversation])
    print(f"完整记忆总字符数: {len(buffer_text)} 字符")
    print(f"节省比例: {(1 - len(summary_text) / len(buffer_text)) * 100:.1f}%")

    # --------------------------------------------------
    # 使用摘要进行新对话
    # --------------------------------------------------
    print("\n--- 基于摘要继续对话 ---\n")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是星辰科技客服小星。以下是之前对话的摘要：\n{summary}"),
        MessagesPlaceholder(variable_name="recent_messages"),
        ("human", "{user_input}"),
    ])

    chain = prompt | chat_llm

    new_input = "我之前说的断连问题，重置之后还是偶尔会断，怎么办？"
    response = chain.invoke({
        "summary": memory.summary,
        "recent_messages": [],
        "user_input": new_input,
    })

    print(f"👤 张三: {new_input}")
    print(f"🤖 小星: {response.content}")
    print("\n  （AI 能从摘要中知道张三买了耳机、有断连问题、已经尝试过重置）")

    print("\n\n--- 三种记忆策略对比 ---")
    print("""
    ┌───────────────────┬──────────┬──────────┬──────────┐
    │        策略        │  记忆完整性 │ Token消耗 │ 适用场景   │
    ├───────────────────┼──────────┼──────────┼──────────┤
    │ BufferMemory      │    ★★★   │    高     │ 短对话    │
    │ SummaryMemory     │    ★★☆   │    低     │ 长对话    │
    │ SlidingWindow     │    ★☆☆   │    中     │ 中等对话  │
    └───────────────────┴──────────┴──────────┴──────────┘
    """)

    print("✓ 第二章示例 2 运行完成！")
    print("  下一课：03_sliding_window.py - 学习滑动窗口记忆")


if __name__ == "__main__":
    main()
