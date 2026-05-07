"""
===============================================
第二章 示例 3：滑动窗口记忆 (SlidingWindowMemory)
===============================================
目标：只保留最近 N 轮对话，平衡记忆完整性和 Token 消耗。

知识点：
    1. 手动实现滑动窗口：保留最近 K 条消息（k*2）
    2. 比 BufferMemory 省 Token，比 SummaryMemory 保留更多细节
    3. 最适合大多数客服场景（通常 5-10 轮对话内能解决问题）

注意：LangChain 1.x 已移除 ConversationBufferWindowMemory，
      改为手动维护消息列表并裁切。

运行方式：
    cd ch02-memory
    python 03_sliding_window.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm_config


class SlidingWindowMemory:
    """手动实现的滑动窗口记忆，等价于已废弃的 ConversationBufferWindowMemory。"""

    def __init__(self, k: int = 2):
        """
        Args:
            k: 保留最近 k 轮对话（每轮包含用户+AI 共 2 条消息）
        """
        self.k = k
        self.messages = []

    def save_context(self, user_input: str, ai_output: str):
        """保存本轮对话并裁剪到窗口大小。"""
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=ai_output))
        # 裁剪：只保留最近 k * 2 条（k 轮 × 每轮 2 条消息）
        self.messages = self.messages[-(self.k * 2):]

    def load_memory_variables(self):
        """返回当前窗口内的记忆。"""
        return {"chat_history": list(self.messages)}


def main():
    print("=" * 60)
    print("  第二章 示例 3：滑动窗口记忆 (SlidingWindowMemory)")
    print("=" * 60)

    config = get_llm_config()
    llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.5,
    )

    # --------------------------------------------------
    # 创建滑动窗口记忆（k=2 表示保留最近 2 轮对话）
    # --------------------------------------------------
    memory = SlidingWindowMemory(k=2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是星辰科技客服小星，回答简洁。注意从对话历史中获取上下文。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}"),
    ])

    chain = prompt | llm

    # --------------------------------------------------
    # 模拟 6 轮对话，观察窗口滑动效果
    # --------------------------------------------------
    print("\n--- 6 轮对话（窗口大小 k=2）---\n")

    conversations = [
        "你好，我叫王五",
        "我上周买了 StarPods Pro",
        "蓝牙连不上手机",        # 第 3 轮：AI 还记得前 2 轮
        "是 iPhone 15 Pro",     # 第 4 轮：AI 记得第 2-3 轮
        "试了重置还是不行",      # 第 5 轮：AI 记得第 3-4 轮
        "那帮我安排换新吧",      # 第 6 轮：AI 记得第 4-5 轮
    ]

    for i, user_input in enumerate(conversations, 1):
        history = memory.load_memory_variables()
        msg_count = len(history.get("chat_history", []))

        print(f"[第 {i} 轮] 记忆中 {msg_count} 条消息")
        print(f"👤 王五: {user_input}")

        response = chain.invoke({
            "chat_history": history.get("chat_history", []),
            "user_input": user_input,
        })
        print(f"🤖 小星: {response.content}")

        memory.save_context(user_input, response.content)

        # 显示窗口内的对话摘要
        hist = memory.load_memory_variables()
        print(f"  📋 窗口内容: ", end="")
        for msg in hist.get("chat_history", []):
            role = "王五" if msg.type == "human" else "小星"
            print(f"[{role}:{msg.content[:10]}...] ", end="")
        print("\n")

    # --------------------------------------------------
    # 验证记忆已被正确裁剪
    # --------------------------------------------------
    print("--- 最终记忆状态 ---")
    history = memory.load_memory_variables()
    print(f"总消息数: {len(history['chat_history'])} 条（预期 4 条，因为 k=2）")
    for msg in history["chat_history"]:
        role = "王五" if msg.type == "human" else "小星"
        print(f"  {role}: {msg.content}")

    print("\n✓ 第二章示例 3 运行完成！")
    print("  下一课：04_multi_turn_bot.py - 完整的多轮客服系统")


if __name__ == "__main__":
    main()
