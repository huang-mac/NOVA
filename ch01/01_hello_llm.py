"""
===============================================
第一章 示例 1：第一个 LLM 调用
===============================================
目标：理解如何使用 LangChain 调用大语言模型。

知识点：
    1. ChatOpenAI 是 LangChain 对 OpenAI Chat API 的封装
    2. HumanMessage / SystemMessage / AIMessage 是三种消息类型
    3. invoke() 方法发送消息并获取回复

运行方式：
    cd ch01-basics
    python 01_hello_llm.py
"""

import sys
import os

# 将 config.py 所在目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import get_llm_config


def main():
    """演示最基本的 LLM 调用。"""

    print("=" * 60)
    print("  第一章 示例 1：第一个 LLM 调用")
    print("=" * 60)

    # --------------------------------------------------
    # 步骤 1：读取配置并初始化 LLM
    # --------------------------------------------------
    config = get_llm_config()
    print(f"\n[配置] 模型: {config['model_name']}")
    print(f"[配置] API 地址: {config['base_url']}")

    llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.7,   # 温度：控制随机性，0=确定性，1=很随机
        max_tokens=500,     # 最大生成 token 数
    )

    # --------------------------------------------------
    # 步骤 2：发送单条消息
    # --------------------------------------------------
    print("\n--- 示例 2.1：单条消息 ---")

    # HumanMessage 代表用户发送的消息
    response = llm.invoke([HumanMessage(content="你好，请用一句话介绍你自己。")])
    print(f"用户: 你好，请用一句话介绍你自己。")
    print(f"AI:   {response.content}")

    # --------------------------------------------------
    # 步骤 3：多条消息对话（模拟聊天记录）
    # --------------------------------------------------
    print("\n--- 示例 2.2：多轮消息 ---")

    # LangChain 通过消息列表模拟完整对话
    # 每条消息都有 role（角色）和 content（内容）
    messages = [
        SystemMessage(content="你是一个友好的助手，回答要简洁。"),
        HumanMessage(content="中国的首都是哪里？"),
        AIMessage(content="中国的首都是北京。"),
        HumanMessage(content="那里有什么著名的景点？"),
    ]

    response = llm.invoke(messages)
    print(f"AI:   {response.content}")

    # --------------------------------------------------
    # 步骤 4：流式输出（打字机效果）
    # --------------------------------------------------
    print("\n--- 示例 2.3：流式输出 ---")
    print("AI:   ", end="", flush=True)

    # stream() 返回一个迭代器，逐块获取生成内容
    for chunk in llm.stream([HumanMessage(content="请用三句话描述春天的景色。")]):
        print(chunk.content, end="", flush=True)

    print("\n")

    # --------------------------------------------------
    # 步骤 5：理解消息类型
    # --------------------------------------------------
    print("\n--- 示例 2.4：三种消息类型说明 ---")
    print("""
    LangChain 中有三种核心消息类型：

    ┌──────────────────┬──────────────────────────────────┐
    │ SystemMessage    │ 系统消息 - 定义 AI 的角色和行为    │
    │                  │ 类似"你是一个客服..."              │
    ├──────────────────┼──────────────────────────────────┤
    │ HumanMessage     │ 用户消息 - 用户说的话              │
    │                  │ 类似"我的订单怎么还没发货？"        │
    ├──────────────────┼──────────────────────────────────┤
    │ AIMessage        │ AI 消息 - AI 之前的回复           │
    │                  │ 用于构建多轮对话历史               │
    └──────────────────┴──────────────────────────────────┘
    """)

    print("\n✓ 第一章示例 1 运行完成！")
    print("  下一课：02_system_prompt.py - 学习如何设计系统提示词")


if __name__ == "__main__":
    main()
