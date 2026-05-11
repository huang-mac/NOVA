"""
===============================================
第八章 示例 3：Execute 阶段 —— 按计划一步步执行
===============================================
目标：按 Planner 生成的计划逐步执行，每一步调用相应的工具并记录结果。

这一步是 Plan-Execute-Replan 的"动手"环节。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_core.tools import tool


# ============================================================
# 模拟数据
# ============================================================
MOCK_ORDERS = {
    "ORD-20240125-001": {
        "order_id": "ORD-20240125-001", "product_name": "StarPods Pro 降噪耳机",
        "price": 899.00, "status": "shipped", "create_time": "2024-01-25 14:30:00",
    },
    "ORD-20240128-002": {
        "order_id": "ORD-20240128-002", "product_name": "StarPods Lite 蓝牙耳机",
        "price": 299.00, "status": "delivered", "create_time": "2024-01-28 20:00:00",
    },
}

MOCK_LOGISTICS = {
    "ORD-20240125-001": {
        "courier": "顺丰速运", "current_location": "北京分拨中心",
        "estimated_delivery": "2024-01-30",
    },
}

MOCK_INVENTORY = {
    "StarPods Pro": {"stock": 156, "status": "有货"},
    "StarPods Lite": {"stock": 0, "status": "缺货"},
    "StarWatch S1": {"stock": 89, "status": "有货"},
}


# ============================================================
# 工具定义
# ============================================================
@tool
def query_order(order_id: str) -> str:
    """查询订单详情。"""
    order = MOCK_ORDERS.get(order_id)
    return json.dumps(order, ensure_ascii=False) if order else json.dumps({"error": f"未找到 {order_id}"}, ensure_ascii=False)

@tool
def query_logistics(order_id: str) -> str:
    """查询物流信息。"""
    info = MOCK_LOGISTICS.get(order_id)
    return json.dumps(info, ensure_ascii=False) if info else json.dumps({"error": "暂无物流信息"}, ensure_ascii=False)

@tool
def query_inventory(product_name: str) -> str:
    """查询库存。"""
    for name, item in MOCK_INVENTORY.items():
        if product_name.lower() in name.lower():
            return json.dumps(item, ensure_ascii=False)
    return json.dumps({"error": f"未找到 {product_name}"}, ensure_ascii=False)

@tool
def check_refund_policy(product_name: str, order_days: int = 0) -> str:
    """查询退款政策。"""
    policy = {"return_period": "7天无理由", "warranty": "1年质保"}
    extra = ""
    if order_days > 7:
        extra = f"\n注意：已下单 {order_days} 天，超出 7 天退货期，但在质保期内。"
    return json.dumps({"policy": policy}, ensure_ascii=False) + extra

TOOLS = [query_order, query_logistics, query_inventory, check_refund_policy]
TOOL_MAP = {t.name: t for t in TOOLS}


# ============================================================
# Executor —— 按计划执行
# ============================================================
def execute_plan(plan: list) -> dict:
    """
    按计划逐步执行，每步调用对应工具并记录结果。

    Args:
        plan: 步骤列表，如 ["查询订单 ORD-xxx", "判断退货政策", ...]

    Returns:
        执行结果字典 {step_index: {"step": ..., "tool": ..., "result": ...}}
    """
    results = {}

    for i, current_step in enumerate(plan):
        print(f"\n  [执行] Step {i + 1}/{len(plan)}: {current_step}")

        # 在步骤描述中匹配工具名（简化版，真实场景由 LLM 的 tool_calls 来做）
        tool_to_call = None
        for tool_name in TOOL_MAP:
            if tool_name.replace("_", " ") in current_step.lower() or tool_name in current_step.lower():
                tool_to_call = tool_name
                break

        if tool_to_call:
            tool_func = TOOL_MAP[tool_to_call]
            print(f"    → 调用工具: {tool_to_call}")
            try:
                result = tool_func.invoke({})
            except Exception as e:
                result = json.dumps({"error": str(e)}, ensure_ascii=False)

            print(f"    ← 结果: {result[:100]}...")
            results[i] = {"step": current_step, "tool": tool_to_call, "result": result}
        else:
            # 不需要工具的步骤（如汇总、生成回复）
            print(f"    → 无需调用工具，直接处理")
            results[i] = {"step": current_step, "tool": None, "result": "已处理"}

    return results


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 3：Execute 阶段 —— 按计划执行")
    print("=" * 60)

    # 模拟 Planner 生成的计划
    test_plans = [
        {
            "desc": "简单物流查询",
            "plan": [
                "查询订单 ORD-20240125-001 的详情",
                "查询订单 ORD-20240125-001 的物流信息",
                "汇总结果，告知用户当前物流状态",
            ],
        },
        {
            "desc": "复杂退货 + 换货场景",
            "plan": [
                "查询订单 ORD-20240125-001 的详情",
                "查询 StarPods Pro 的退款政策（已购 10 天）",
                "查询 StarPods Lite 的库存信息",
                "汇总退货和换货方案，告知用户",
            ],
        },
    ]

    for test in test_plans:
        print(f"\n{'=' * 60}")
        print(f"测试：{test['desc']}")
        print(f"计划：{test['plan']}")
        print("-" * 60)

        results = execute_plan(test["plan"])

        print(f"\n  执行结果汇总：")
        for idx, r in results.items():
            tool_str = f" [{r['tool']}]" if r["tool"] else ""
            print(f"    Step {idx + 1}{tool_str}: {r['result'][:80]}...")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. Executor 是「按图施工」，按计划逐步推进")
    print("  2. 每步执行完，结果存入 State，供后续步骤使用")
    print("  3. 工具匹配可以用关键词匹配（简化版）或 LLM tool_calls（完整版）")
    print("  4. 某些步骤不需要工具（如汇总），直接标记为已处理")
    print("  5. 如果某步出错，需要标记以便 Replan 阶段处理")
    print("=" * 60)
