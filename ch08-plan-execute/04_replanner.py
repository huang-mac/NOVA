"""
===============================================
第八章 示例 4：Replan 阶段 —— 发现不对就调整
===============================================
目标：执行到一半发现情况变了（比如工具报错、结果不符预期），
     AI 能根据中间结果重新调整计划。

这是 Plan-Execute-Replan 最精妙的部分——自我纠正能力。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import get_llm


# ============================================================
# 模拟数据
# ============================================================
MOCK_ORDERS = {
    "ORD-20240125-001": {
        "order_id": "ORD-20240125-001", "product_name": "StarPods Pro 降噪耳机",
        "price": 899.00, "status": "refund_processing", "create_time": "2024-01-25 14:30:00",
    },
    "ORD-20240128-002": {
        "order_id": "ORD-20240128-002", "product_name": "StarPods Lite 蓝牙耳机",
        "price": 299.00, "status": "delivered", "create_time": "2024-01-28 20:00:00",
    },
}

MOCK_LOGISTICS = {
    "ORD-20240125-001": {"error": "暂无物流信息"},  # 注意：这笔订单在退款中，没有物流
}


# ============================================================
# 工具定义
# ============================================================
def query_order(order_id: str) -> str:
    """查询订单详情。"""
    order = MOCK_ORDERS.get(order_id)
    return json.dumps(order, ensure_ascii=False) if order else json.dumps({"error": f"未找到 {order_id}"}, ensure_ascii=False)

def query_logistics(order_id: str) -> str:
    """查询物流信息。"""
    info = MOCK_LOGISTICS.get(order_id)
    return json.dumps(info, ensure_ascii=False) if info else json.dumps({"error": "暂无物流信息"}, ensure_ascii=False)


# ============================================================
# Replanner —— 根据中间结果调整计划
# ============================================================
REPLANNER_PROMPT = """你是一个任务规划专家。之前的执行过程中出现了一些情况，需要你根据中间结果调整计划。

之前的计划：
{original_plan}

已执行的结果：
{execution_results}

请根据以上信息：
1. 分析哪些步骤已经成功完成
2. 分析哪些步骤出现了问题或不符合预期
3. 制定剩余步骤的调整计划

请用 JSON 格式输出：
{{"analysis": "简要分析", "completed_steps": [已完成步骤的索引], "new_steps": [剩余需要执行的步骤]}}"""


def replan(original_plan: list, execution_results: dict) -> dict:
    """
    根据执行过程中的中间结果，重新调整计划。

    Args:
        original_plan: 原始计划步骤列表
        execution_results: 已执行的步骤结果 {step_index: {"step": ..., "tool": ..., "result": ...}}

    Returns:
        调整后的计划信息 {"analysis": ..., "completed_steps": [...], "new_steps": [...]}
    """
    llm = get_llm(temperature=0)

    # 格式化已执行的结果
    results_text = ""
    for idx, result in execution_results.items():
        status = "成功" if "error" not in result.get("result", "") else "失败"
        results_text += f"\nStep {idx + 1} [{status}]: {result['step']}\n  结果: {result['result']}\n"

    messages = [
        SystemMessage(content=REPLANNER_PROMPT.format(
            original_plan=json.dumps(original_plan, ensure_ascii=False, indent=2),
            execution_results=results_text,
        )),
        HumanMessage(content="请根据以上执行结果调整计划。"),
    ]

    response = llm.invoke(messages)
    print(f"\n  [Replanner] 分析：\n{response.content}\n")

    # 解析
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {
            "analysis": "解析失败，建议人工检查",
            "completed_steps": list(execution_results.keys()),
            "new_steps": [],
        }


# ============================================================
# 简化的执行器（用于演示 Replan 流程）
# ============================================================
def simple_execute_step(step: str) -> dict:
    """执行单个步骤并返回结果。"""
    result_str = ""
    if "订单" in step and "物流" not in step:
        # 模拟查询订单
        if "ORD-20240125-001" in step:
            result_str = json.dumps(MOCK_ORDERS["ORD-20240125-001"], ensure_ascii=False)
        else:
            result_str = json.dumps({"error": "未找到订单"}, ensure_ascii=False)
    elif "物流" in step:
        # 模拟查询物流
        if "ORD-20240125-001" in step:
            result_str = json.dumps(MOCK_LOGISTICS["ORD-20240125-001"], ensure_ascii=False)
        else:
            result_str = json.dumps({"error": "暂无物流信息"}, ensure_ascii=False)
    else:
        result_str = "已处理（无需工具）"

    return result_str


# ============================================================
# 完整 Plan-Execute-Replan 流程演示
# ============================================================
def run_plan_execute_replan(user_question: str, initial_plan: list):
    """
    演示 Plan → Execute（遇到问题） → Replan → Execute 的完整流程。
    """
    print(f"\n{'=' * 60}")
    print(f"用户问题：{user_question}")
    print(f"初始计划：{initial_plan}")
    print("-" * 60)

    plan = initial_plan
    results = {}
    step_index = 0

    # 第一轮执行
    print("\n  --- 第一轮执行 ---")
    while step_index < len(plan):
        step = plan[step_index]
        result_str = simple_execute_step(step)

        has_error = "error" in result_str
        print(f"  [执行] Step {step_index + 1}: {step}")
        print(f"    结果: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")

        results[step_index] = {"step": step, "result": result_str}

        if has_error:
            print(f"    ⚠ 发现问题！触发 Replan")
            break

        step_index += 1

    # 检查是否需要 Replan
    if step_index < len(plan):
        print("\n  --- Replan 阶段 ---")
        replan_result = replan(plan, results)
        print(f"  [Replanner] 分析：{replan_result.get('analysis', '无')}")

        new_steps = replan_result.get("new_steps", [])
        if new_steps:
            print(f"  [Replanner] 调整后的计划：{new_steps}")

            # 第二轮执行
            print("\n  --- 第二轮执行（调整后） ---")
            for i, step in enumerate(new_steps):
                result_str = simple_execute_step(step)
                new_idx = len(results)
                results[new_idx] = {"step": step, "result": result_str}
                print(f"  [执行] Step {new_idx + 1}: {step}")
                print(f"    结果: {result_str[:100]}{'...' if len(result_str) > 100 else ''}")
        else:
            print(f"  [Replanner] 无需进一步操作")

    # 汇总
    print(f"\n  --- 执行完成 ---")
    print(f"  总步骤数：{len(results)}")
    for idx, r in results.items():
        print(f"    Step {idx + 1}: {r['step']} → {r['result'][:60]}...")


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 4：Replan 阶段 —— 发现不对就调整")
    print("=" * 60)

    # 场景：用户要查物流，但订单已经在退款中（没有物流信息了）
    run_plan_execute_replan(
        "帮我查一下订单 ORD-20240125-001 的物流到哪了",
        [
            "查询订单 ORD-20240125-001 的详情",
            "查询订单 ORD-20240125-001 的物流信息",
            "汇总物流状态，告知用户",
        ],
    )

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. Replan 的触发条件：执行结果包含 error、结果与预期不符")
    print("  2. Replan 不是推倒重来，而是在已有结果基础上调整")
    print("  3. Replanner 能分析哪些步骤已完成、哪些需要调整")
    print("  4. 调整后的计划是增量的，已完成的不需要重复执行")
    print("  5. 设置最大重规划次数（如 max_replans=3）防止无限循环")
    print("=" * 60)
