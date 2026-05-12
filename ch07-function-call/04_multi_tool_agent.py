"""
===============================================
第七章 示例 4：多 Tool 协作 Agent
===============================================
目标：实现一个带多个客服工具的 Agent，LLM 自动判断需要调用几个工具，
     并综合多个工具的结果生成最终回复。

这个示例是本章的核心——展示了 Function Call 在真实客服场景中的完整流程。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
from config import get_llm


# 导入工具定义
from langchain_core.tools import tool


# ============================================================
# 模拟数据
# ============================================================
MOCK_ORDERS = {
    "ORD-20240125-001": {
        "order_id": "ORD-20240125-001",
        "user_id": "U10086",
        "product_name": "StarPods Pro 降噪耳机",
        "quantity": 1,
        "price": 899.00,
        "status": "shipped",
        "create_time": "2024-01-25 14:30:00",
    },
    "ORD-20240128-002": {
        "order_id": "ORD-20240128-002",
        "user_id": "U10086",
        "product_name": "StarWatch S1 智能手表",
        "quantity": 2,
        "price": 2598.00,
        "status": "pending",
        "create_time": "2024-01-28 20:00:00",
    },
}

MOCK_LOGISTICS = {
    "ORD-20240125-001": {
        "order_id": "ORD-20240125-001",
        "courier": "顺丰速运",
        "tracking_number": "SF1234567890",
        "current_location": "已到达北京分拨中心",
        "estimated_delivery": "2024-01-28",
    },
}

MOCK_INVENTORY = {
    "StarPods Pro": {"product_name": "StarPods Pro 降噪耳机", "price": 899.00, "stock": 156, "status": "有货"},
    "StarPods Lite": {"product_name": "StarPods Lite 蓝牙耳机", "price": 299.00, "stock": 0, "status": "缺货"},
    "StarWatch S1": {"product_name": "StarWatch S1 智能手表", "price": 1299.00, "stock": 89, "status": "有货"},
}


# ============================================================
# 工具定义
# ============================================================
@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单详情，包括商品名称、金额、状态、下单时间等。
    当用户询问订单信息、订单状态、买了什么时使用。

    Args:
        order_id: 订单号，例如 ORD-20240125-001
    """
    order = MOCK_ORDERS.get(order_id)
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return json.dumps({"error": f"未找到订单 {order_id}，请确认订单号是否正确。"}, ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """查询订单的物流信息，包括快递公司、当前所在位置、预计送达时间等。
    当用户问包裹到哪了、什么时候到、快递单号是多少时使用。

    Args:
        order_id: 订单号
    """
    info = MOCK_LOGISTICS.get(order_id)
    if info:
        return json.dumps(info, ensure_ascii=False, indent=2)
    return json.dumps({"error": f"订单 {order_id} 暂无物流信息，可能尚未发货。"}, ensure_ascii=False)


@tool
def query_inventory(product_name: str) -> str:
    """查询商品库存信息，包括价格、库存数量、是否有货。
    当用户问某个商品有没有货、多少钱、库存多少时使用。

    Args:
        product_name: 商品名称，例如 StarPods Pro、StarWatch S1
    """
    for name, item in MOCK_INVENTORY.items():
        if product_name.lower() in name.lower() or name.lower() in product_name.lower():
            return json.dumps(item, ensure_ascii=False, indent=2)
    return json.dumps({"error": f"未找到商品 '{product_name}'"}, ensure_ascii=False)


@tool
def check_refund_policy(product_name: str, order_days: int = 0) -> str:
    """查询退款和退货政策。
    当用户问能不能退货、退换货规则是什么时使用。

    Args:
        product_name: 商品名称
        order_days: 商品下单了多少天（如果用户提到了就传入）
    """
    policy = {
        "return_period": "7 天无理由退货（未拆封）",
        "warranty": "1 年质保",
        "refund_time": "收到退货后 3 个工作日退款",
    }
    extra = ""
    if order_days > 0:
        if order_days <= 7:
            extra = f"\n该商品已下单 {order_days} 天，仍在退货期内。"
        else:
            extra = f"\n该商品已下单 {order_days} 天，已超出 7 天退货期，但质保期内可免费维修。"
    return json.dumps({"product": product_name, "policy": policy}, ensure_ascii=False, indent=2) + extra


# 所有工具
ALL_TOOLS = [query_order, query_logistics, query_inventory, check_refund_policy]

# 工具名到工具对象的映射
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ============================================================
# 系统提示词
# ============================================================
SYSTEM_PROMPT = """你是一名专业的星辰科技客服人员。你可以使用以下工具来帮助用户：
- query_order: 查询订单详情
- query_logistics: 查询物流信息
- query_inventory: 查询商品库存
- check_refund_policy: 查询退款政策

## 使用规则
1. 如果用户提供了订单号，直接查询，不要反问
2. 如果用户没有提供订单号但明显需要查询，礼貌索要
3. 一个问题可能需要调用多个工具，请根据情况判断
4. 查询到结果后，用简洁友好的语言总结，不要返回原始 JSON
5. 工具查询失败时，如实告知用户并给出建议
6. 涉及退款/换货操作时，告知用户流程，引导去 APP 申请

## 回复风格
- 简洁、友好、专业
- 不要废话，直接给用户想要的信息
- 遇到问题主动提供解决方案"""


# ============================================================
# 核心：带多 Tool 的客服 Agent
# ============================================================
class ToolAgent:
    """
    带多 Tool 的客服 Agent。

    完整的 Function Call 循环：
    1. LLM 判断是否需要调用工具
    2. 如果需要，执行工具调用
    3. 把工具结果喂回 LLM
    4. LLM 可能决定再调用其他工具（循环）
    5. 最终生成自然语言回复
    """

    def __init__(self, max_tool_rounds: int = 5):
        """
        Args:
            max_tool_rounds: 最大工具调用轮数，防止无限循环
        """
        self.llm = get_llm(temperature=0)
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.max_tool_rounds = max_tool_rounds

    def chat(self, user_message: str) -> dict:
        """
        处理用户消息，返回回复和调用记录。

        Args:
            user_message: 用户的输入消息

        Returns:
            dict: {"reply": str, "tool_calls": list, "total_rounds": int}
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        tool_calls_log = []  # 记录所有工具调用
        total_rounds = 0

        while total_rounds < self.max_tool_rounds:
            total_rounds += 1
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)  # 把 AI 的回复加入历史

            # 如果 LLM 没有调用工具，说明它已经生成了最终回复
            if not response.tool_calls:
                break

            # 执行所有工具调用
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                print(f"  [Round {total_rounds}] 调用工具: {tool_name}({tool_args})")

                # 查找并执行工具
                tool_func = TOOL_MAP.get(tool_name)
                if tool_func:
                    try:
                        result = tool_func.invoke(tool_args)
                    except Exception as e:
                        result = json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)
                else:
                    result = json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

                # 记录调用日志
                tool_calls_log.append({
                    "round": total_rounds,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result[:100] + "..." if len(result) > 100 else result,
                })

                # 把工具结果作为 ToolMessage 加入历史
                messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call_id)
                )

        # 提取最终回复
        final_reply = response.content if response.content else "抱歉，我暂时无法处理您的问题，已为您转接人工客服。"

        return {
            "reply": final_reply,
            "tool_calls": tool_calls_log,
            "total_rounds": total_rounds,
        }


# ============================================================
# 测试场景
# ============================================================
def run_test(question: str, description: str):
    """运行一个测试场景。"""
    print(f"\n{'=' * 60}")
    print(f"测试：{description}")
    print(f"用户：{question}")
    print("-" * 60)

    agent = ToolAgent()
    result = agent.chat(question)

    print(f"\nAI 回复：{result['reply']}")
    print(f"\n工具调用记录（共 {result['total_rounds']} 轮）：")
    for log in result["tool_calls"]:
        print(f"  Round {log['round']}: {log['tool']}({log['args']})")
        print(f"    结果: {log['result']}")
    if not result["tool_calls"]:
        print("  （无工具调用）")


if __name__ == "__main__":
    print("=" * 60)
    print("第七章 示例 4：多 Tool 协作 Agent")
    print("=" * 60)

    # 测试 1：单个工具调用
    run_test(
        "帮我查一下订单 ORD-20240125-001 的状态",
        "查订单（单工具）"
    )

    # 测试 2：需要两个工具（先查订单状态 → 再查物流）
    run_test(
        "我的订单 ORD-20240125-001 到哪了？什么时候能到？",
        "查物流（需要判断订单已发货才查物流）"
    )

    # 测试 3：查库存
    run_test(
        "StarPods Pro 还有没有货？",
        "查库存（单工具）"
    )

    # 测试 4：复杂问题（可能需要多个工具）
    run_test(
        "我的耳机订单 ORD-20240125-001 到了之后如果不喜欢能退吗？现在 StarPods Lite 有没有货可以换？",
        "退货政策 + 查库存（多工具协作）"
    )

    # 测试 5：不需要工具的问题
    run_test(
        "你好，你们的客服电话是多少？",
        "不需要工具的问题"
    )

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. LLM 会自动判断需要调用几个工具")
    print("  2. 多轮工具调用是自动循环的，不需要你手动控制")
    print("  3. 设置 max_tool_rounds 防止无限循环")
    print("  4. 每次工具结果都要作为 ToolMessage 喂回 LLM")
    print("=" * 60)
