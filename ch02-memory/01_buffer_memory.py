"""
===============================================
第二章 示例 1：完整对话记忆 (BufferMemory)
===============================================
目标：手动管理完整对话历史，实现多轮对话。

知识点：
    1. 使用列表存储所有对话记录
    2. 手动 save_context / load 历史消息
    3. 使用 RunnableWithMessageHistory 实现自动记忆管理

注意：LangChain 1.x 已移除 langchain.memory 模块，
      改为手动管理消息列表，逻辑更清晰可控。

运行方式：
    cd ch02-memory
    python 01_buffer_memory.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm_config


def main():
    print("=" * 60)
    print("  第二章 示例 1：完整对话记忆 (BufferMemory)")
    print("=" * 60)

    config = get_llm_config()
    llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.5,
    )

    # --------------------------------------------------
    # 方式 1：手动管理记忆（用列表存储消息）
    # --------------------------------------------------
    print("\n--- 方式 1：手动管理记忆 ---\n")

    # 用列表代替 ConversationBufferMemory
    chat_history = []

    # 构建对话提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是星辰科技的客服小星，回答简洁友好。"),
        MessagesPlaceholder(variable_name="chat_history"),  # 记忆插入点
        ("human", "{user_input}"),
    ])

    # 模拟多轮对话
    demo_conversations = [
        "我叫张三，我的订单号是 ORD-20240115-001",
        "我上周买的 StarPods Pro 耳机有杂音",
        "已经买了两周了，在保修期内吗？",
    ]

    for i, user_input in enumerate(demo_conversations, 1):
        print(f"[第 {i} 轮]")
        print(f"👤 用户: {user_input}")

        # 1. 构建带历史的完整消息列表
        messages = prompt.format_messages(
            chat_history=chat_history,
            user_input=user_input,
        )

        # 2. 调用 LLM 生成回复
        response = llm.invoke(messages)
        print(f"🤖 小星: {response.content}")

        # 3. 将本轮对话保存到历史（等价于 memory.save_context）
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))
        print()

    # 查看当前记忆中的所有对话
    print("--- 当前记忆内容 ---")
    for msg in chat_history:
        role = "用户" if msg.type == "human" else "小星"
        print(f"  {role}: {msg.content}")
    print(f"\n  共 {len(chat_history)} 条消息")

    # --------------------------------------------------
    # 方式 2：使用 RunnableWithMessageHistory（推荐）
    # --------------------------------------------------
    print("\n\n--- 方式 2：自动管理记忆（推荐方式） ---\n")

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory

    # 使用 LCEL 构建链
    chain = prompt | llm

    # 创建会话历史存储（每个 session_id 对应一个独立的对话历史）
    session_store = {}

    def get_session_history(session_id: str):
        """根据 session_id 获取或创建对话历史。"""
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    # 用 RunnableWithMessageHistory 包装链，自动管理对话历史
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_input",      # 输入消息的 key
        history_messages_key="chat_history",   # 历史消息的 key
    )

    # 使用 session_id 区分不同用户/会话
    session_id = "user-zhangsan-001"

    # 对话时自动保存和加载历史
    conversations = [
        "我叫李四，订单号是 ORD-20240120-002",
        "我的智能手表 StarWatch X 屏幕不亮了",
    ]

    for user_input in conversations:
        response = chain_with_history.invoke(
            {"user_input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"👤 李四: {user_input}")
        print(f"🤖 小星: {response.content}\n")

    # 验证不同 session 之间的记忆是隔离的
    print("--- 验证会话隔离 ---")
    other_session = "user-wangwu-002"
    response = chain_with_history.invoke(
        {"user_input": "你还记得我叫什么吗？"},
        config={"configurable": {"session_id": other_session}},
    )
    print(f"👤 新用户: 你还记得我叫什么吗？")
    print(f"🤖 小星: {response.content}")
    print("  （新会话中 AI 不记得张三/李四的信息 ✓）")

    print("\n✓ 第二章示例 1 运行完成！")
    print("  下一课：02_summary_memory.py - 学习摘要记忆")


if __name__ == "__main__":
    main()
