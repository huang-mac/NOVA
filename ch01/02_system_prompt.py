"""
===============================================
第一章 示例 2：系统提示词设计
===============================================
目标：学习如何通过 System Prompt 设计客服 AI 的角色、行为规范和约束。

知识点：
    1. System Prompt（系统提示词）定义 AI 的"人设"和工作规则
    2. 好的系统提示词 = 角色定义 + 行为规范 + 输出格式 + 边界约束
    3. 使用 ChatPromptTemplate 模板化管理提示词

运行方式：
    cd ch01
    python 02_system_prompt.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import get_llm_config


def main():
    print("=" * 60)
    print("  第一章 示例 2：系统提示词设计")
    print("=" * 60)

    config = get_llm_config()
    llm = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=0.5,  # 客服场景温度偏低，保证回答稳定
    )

    # --------------------------------------------------
    # 步骤 1：对比有无系统提示词的效果
    # --------------------------------------------------
    print("\n--- 示例 1：有无系统提示词的对比 ---\n")

    user_input = "你们的退款流程是什么？"

    # 没有系统提示词的回复
    print("[无 System Prompt]")
    response = llm.invoke([HumanMessage(content=user_input)])
    print(f"AI: {response.content}\n")

    # 有系统提示词的回复
    print("[有 System Prompt - 客服角色]")
    system_prompt = """你是「星辰科技」的智能客服助手。请遵循以下规范：

## 角色定位
你代表星辰科技官方，态度专业、友善、有耐心。

## 行为规范
1. 先理解用户问题，再给出准确回答
2. 不确定的信息要诚实告知"我需要核实一下"
3. 回答要简洁清晰，避免冗长

## 退款政策
- 7天内无理由退款
- 退款将在3-5个工作日内到账
- 请联系客服提供订单号

## 禁止事项
- 不得承诺无法兑现的服务
- 不得讨论公司内部信息
- 不得对用户态度恶劣
"""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])
    print(f"AI: {response.content}\n")

    # --------------------------------------------------
    # 步骤 2：使用 ChatPromptTemplate 管理提示词
    # --------------------------------------------------
    print("--- 示例 2：使用 Prompt 模板 ---\n")

    # ChatPromptTemplate 将消息模板化，支持变量替换
    # 这是 LangChain 推荐的提示词管理方式
    prompt_template = ChatPromptTemplate.from_messages([
        # SystemMessage: 定义 AI 角色
        ("system", "你是{company_name}的客服代表。回答要{style}。当前时间：{time}。"),
        # MessagesPlaceholder: 占位符，后续可填入历史消息
        MessagesPlaceholder(variable_name="chat_history"),
        # HumanMessage: 用户的输入
        ("human", "{user_input}"),
    ])

    # 使用 .format_messages() 填充模板变量
    messages = prompt_template.format_messages(
        company_name="星辰科技",
        style="专业、简洁、有温度",
        time="2024年1月15日 下午3点",
        chat_history=[],  # 暂时没有历史消息，下一章会用到
        user_input="我上周买的耳机有杂音，怎么办？",
    )

    response = llm.invoke(messages)
    print(f"公司: 星辰科技 | 风格: 专业、简洁、有温度")
    print(f"用户: 我上周买的耳机有杂音，怎么办？")
    print(f"AI:   {response.content}\n")

    # --------------------------------------------------
    # 步骤 3：使用 LCEL 链式调用（LangChain Expression Language）
    # --------------------------------------------------
    print("--- 示例 3：LCEL 链式调用 ---\n")

    # LCEL 是 LangChain 的声明式语法
    # 用管道符 | 将 prompt 和 llm 串联成一条链
    chain = prompt_template | llm

    # 直接调用链，传入模板变量
    response = chain.invoke({
        "company_name": "星辰科技",
        "style": "轻松活泼",
        "time": "2024年1月15日 下午3点",
        "chat_history": [],
        "user_input": "你们的产品有什么优势？",
    })
    print(f"公司: 星辰科技 | 风格: 轻松活泼")
    print(f"用户: 你们的产品有什么优势？")
    print(f"AI:   {response.content}\n")

    # --------------------------------------------------
    # 步骤 4：系统提示词编写技巧总结
    # --------------------------------------------------
    print("--- 系统提示词编写技巧 ---")
    print("""
    好的系统提示词结构（客服场景）：

    ┌─────────────────────────────────────────────┐
    │  1. 角色定义 - 你是谁？                       │
    │     "你是 XX 公司的智能客服..."               │
    ├─────────────────────────────────────────────┤
    │  2. 知识范围 - 你知道什么？                     │
    │     产品信息、政策、常见问题 FAQ                │
    ├─────────────────────────────────────────────┤
    │  3. 行为规范 - 你该怎么回答？                   │
    │     语气要求、回答格式、长度限制                │
    ├─────────────────────────────────────────────┤
    │  4. 边界约束 - 你不能做什么？                   │
    │     不能泄露隐私、不能承诺超出权限的事            │
    ├─────────────────────────────────────────────┤
    │  5. 输出格式 - 回复应该长什么样？                │
    │     JSON / 纯文本 / 特定模板                   │
    └─────────────────────────────────────────────┘

    关键原则：
    - 具体 > 模糊："语气专业" 比 "好听" 更有效
    - 给例子 > 只说规则：直接给出好/坏回复示例
    - 分层组织：用 Markdown 标题分块，便于模型理解
    """)

    print("\n✓ 第一章示例 2 运行完成！")
    print("  下一课：03_basic_chatbot.py - 实现完整的基础客服对话机器人")


if __name__ == "__main__":
    main()
