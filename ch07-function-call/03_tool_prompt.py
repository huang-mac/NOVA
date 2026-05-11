"""
===============================================
第八章 示例 3：Tool 场景的 Prompt 工程
===============================================
目标：演示如何通过 Prompt 优化 LLM 使用 Tool 的效果。

工具定义得再好，如果 LLM 不知道"怎么用"，效果也打折。
这一章讲三个关键技巧：
1. 系统提示词中明确工具使用规则
2. 处理工具调用失败的情况
3. 引导 LLM 综合多个工具结果生成自然语言回复
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from config import get_llm
from langchain_openai import ChatOpenAI


# 导入上一节定义的工具
from langchain_core.tools import tool


def load_mock_orders():
    return {
        "ORD-20240125-001": {
            "order_id": "ORD-20240125-001",
            "product_name": "StarPods Pro 降噪耳机",
            "quantity": 1,
            "price": 899.00,
            "status": "shipped",
            "create_time": "2024-01-25 14:30:00",
        },
    }


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单详情。当用户询问订单状态时使用。

    Args:
        order_id: 订单号，格式为 ORD-xxxxxxxx-xxx
    """
    orders = load_mock_orders()
    order = orders.get(order_id)
    if order:
        return json.dumps(order, ensure_ascii=False)
    return json.dumps({"error": f"未找到订单 {order_id}"}, ensure_ascii=False)


tools = [query_order]


# ============================================================
# 1. 好的系统提示词 vs 坏的系统提示词
# ============================================================

# 不好的系统提示词 —— 太简单，LLM 行为不可控
BAD_SYSTEM_PROMPT = """你是一名客服，请回答用户的问题。"""

# 好的系统提示词 —— 明确了工具使用规则和边界
GOOD_SYSTEM_PROMPT = """你是一名专业的星辰科技客服人员。你可以使用工具来查询订单信息。

## 工具使用规则
1. 当用户提供了订单号，直接使用 query_order 工具查询，不要让用户等待
2. 当用户没有提供订单号，礼貌地请用户提供（示例回复："好的，请您提供一下订单号，格式通常是 ORD-xxxxxxxx-xxx"）
3. 查询到结果后，用简洁友好的自然语言总结给用户，不要直接返回原始 JSON 数据
4. 如果查询失败（订单号不存在），如实告知用户，不要编造结果
5. 涉及退款、换货等操作时，引导用户在 APP 中申请，你只负责查询和告知

## 回复风格
- 语气亲切但专业
- 回复尽量简短，用户不想看长篇大论
- 遇到问题主动提供解决方案"""


def demo_bad_prompt():
    """不好的系统提示词效果。"""
    print("=" * 40)
    print("不好的系统提示词效果")
    print("=" * 40)

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=BAD_SYSTEM_PROMPT),
        HumanMessage(content="我的订单 ORD-20240125-001 怎么还没到？"),
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print("✓ LLM 决定调用工具（正确）")
        # 执行工具
        tool_result = query_order.invoke(response.tool_calls[0]["args"])
        print(f"  工具返回：{tool_result[:80]}...")

        # 喂回 LLM
        tool_msg = ToolMessage(content=tool_result, tool_call_id=response.tool_calls[0]["id"])
        final = llm_with_tools.invoke(messages + [response, tool_msg])
        print(f"  回复：{final.content}")
    else:
        print(f"✗ LLM 没有调用工具，直接回复：{response.content}")
        print("  （问题：没有明确指引，LLM 可能不知道该用工具）")


def demo_good_prompt():
    """好的系统提示词效果。"""
    print("\n" + "=" * 40)
    print("好的系统提示词效果")
    print("=" * 40)

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=GOOD_SYSTEM_PROMPT),
        HumanMessage(content="我的订单 ORD-20240125-001 怎么还没到？"),
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print("✓ LLM 决定调用工具")
        tool_result = query_order.invoke(response.tool_calls[0]["args"])
        tool_msg = ToolMessage(content=tool_result, tool_call_id=response.tool_calls[0]["id"])
        final = llm_with_tools.invoke(messages + [response, tool_msg])
        print(f"  回复：{final.content}")
    else:
        print(f"✗ LLM 没有调用工具：{response.content}")


# ============================================================
# 2. 处理工具调用失败
# ============================================================
def demo_error_handling():
    """演示 LLM 处理工具调用失败（订单不存在）的场景。"""
    print("\n" + "=" * 40)
    print("工具调用失败处理")
    print("=" * 40)

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=GOOD_SYSTEM_PROMPT),
        HumanMessage(content="帮我查一下订单 ORD-99999999-999"),
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_result = query_order.invoke(response.tool_calls[0]["args"])
        print(f"  工具返回（含错误）：{tool_result}")

        # 即使工具返回了错误信息，也要喂回给 LLM，让它生成友好回复
        tool_msg = ToolMessage(content=tool_result, tool_call_id=response.tool_calls[0]["id"])
        final = llm_with_tools.invoke(messages + [response, tool_msg])
        print(f"  LLM 的回复（处理了错误）：{final.content}")


# ============================================================
# 3. 用户没给订单号，LLM 应该索要而不是瞎调
# ============================================================
def demo_ask_for_info():
    """演示 LLM 在信息不足时主动询问。"""
    print("\n" + "=" * 40)
    print("信息不足时主动询问")
    print("=" * 40)

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=GOOD_SYSTEM_PROMPT),
        HumanMessage(content="我想查一下我的耳机发货了没"),
    ]

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"✗ LLM 调用了工具（不应该，因为没有订单号）")
        print(f"  参数：{response.tool_calls[0]['args']}")
    else:
        print(f"✓ LLM 没有调用工具，而是索要信息：{response.content}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 3：Tool 场景的 Prompt 工程")
    print("=" * 60)

    # 对比好提示词 vs 坏提示词
    demo_bad_prompt()
    demo_good_prompt()

    # 错误处理
    demo_error_handling()

    # 主动询问
    demo_ask_for_info()

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. 系统提示词要明确告诉 LLM：什么时候用工具、用什么参数、怎么处理结果")
    print("  2. 工具返回错误时，也要喂回 LLM，让它生成友好回复")
    print("  3. 信息不足时，LLM 应该主动询问，而不是瞎调工具")
    print("  4. Prompt 越具体，LLM 的行为越可预测")
    print("=" * 60)
