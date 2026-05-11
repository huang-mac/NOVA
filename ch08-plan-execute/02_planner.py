"""
===============================================
第八章 示例 2：Plan 阶段 —— 让 AI 自己制定计划
===============================================
目标：用 LLM 把用户的复杂客服问题拆解成一个分步骤的执行计划。

这是 Plan-Execute-Replan 的第一步：AI 不急着回答，先想清楚要做什么。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


# ============================================================
# 模拟数据（同完整版）
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


# ============================================================
# Planner 的 System Prompt
# ============================================================
PLANNER_PROMPT = """你是一个任务规划专家。分析用户的客服问题，制定一个清晰的执行计划。

可用工具：
- query_order(order_id): 查询订单详情（返回订单状态、商品、价格等）
- query_logistics(order_id): 查询物流信息（返回快递公司、当前位置、预计到达）
- query_inventory(product_name): 查询商品库存（返回库存数量和状态）
- check_refund_policy(product_name, order_days): 查询退款政策（返回退货期和质保信息）

规则：
1. 每步是一个具体的、可执行的动作
2. 标注需要调用的工具和参数
3. 如果某步依赖上一步结果，标注 "depends_on: step X"
4. 最后一步应该是"汇总结果，生成最终回复"
5. 计划尽量简洁，避免不必要的步骤

请用 JSON 格式输出：
{"steps": ["步骤1描述", "步骤2描述", ...]}"""


# ============================================================
# Planner 函数
# ============================================================
def create_plan(user_message: str) -> list:
    """
    让 LLM 分析用户问题，生成执行计划。

    Args:
        user_message: 用户的原始问题

    Returns:
        步骤列表，如 ["查询订单 ORD-xxx 的详情", "根据订单信息判断退货政策", ...]
    """
    llm = get_llm(temperature=0)

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    print(f"\n  [Planner] 原始输出：\n{response.content}\n")

    # 解析计划
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])
    except (json.JSONDecodeError, IndexError):
        # 解析失败时，尝试按行分割
        steps = [
            line.strip().lstrip("0123456789.) ")
            for line in response.content.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]

    return steps


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 2：Plan 阶段 —— 让 AI 制定执行计划")
    print("=" * 60)

    test_cases = [
        "查一下订单 ORD-20240125-001 的物流",
        "我买了 StarPods Pro，订单号 ORD-20240125-001，到了之后想退货，现在 StarPods Lite 有没有货可以换？",
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"测试 {i}：{question}")
        print("-" * 60)

        plan = create_plan(question)

        print(f"  执行计划（共 {len(plan)} 步）：")
        for j, step in enumerate(plan, 1):
            print(f"    {j}. {step}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. Planner 的输入是用户问题，输出是结构化的步骤列表")
    print("  2. 用 JSON 格式输出便于后续解析和执行")
    print("  3. 步骤之间可以有依赖关系（depends_on）")
    print("  4. 计划不需要 LLM 立刻执行，而是存到 State 里等 Executor 处理")
    print("  5. 养成加 fallback 解析的习惯（JSON 解析失败时按行分割）")
    print("=" * 60)
