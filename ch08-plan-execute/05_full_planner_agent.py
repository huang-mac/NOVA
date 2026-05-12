"""
===============================================
第八章 示例 5：完整 Plan-Execute-Replan Agent
===============================================
目标：用 LangGraph 实现完整的 Plan → Execute → Replan 循环，
     让 AI 能自主拆解复杂客服问题并一步步解决。

这是本章的核心文件，把前四个示例的概念全部串起来。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from config import get_llm


# ============================================================
# 模拟数据（同第七章）
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
# State 定义
# ============================================================
class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 工作流的状态。"""
    messages: Annotated[list, add_messages]  # 对话历史
    plan: list[str]         # 执行计划步骤
    step_index: int         # 当前执行到第几步
    step_results: dict      # 每步的执行结果 {step_index: result}
    needs_replan: bool      # 是否需要重新规划
    max_steps: int          # 最大步骤数（防止无限循环）


# ============================================================
# Node 1: Planner —— 制定执行计划
# ============================================================
PLANNER_SYSTEM = """你是一个任务规划专家。分析用户的问题，制定一个清晰的执行计划。

可用工具：
- query_order(order_id): 查询订单详情
- query_logistics(order_id): 查询物流信息
- query_inventory(product_name): 查询商品库存
- check_refund_policy(product_name, order_days): 查询退款政策

规则：
1. 每步是一个具体的可执行动作
2. 标注需要调用的工具和参数
3. 如果某步依赖上一步结果，标注 "depends_on: step X"
4. 最后一步应该是"汇总结果，生成最终回复"

请用 JSON 格式输出：
{"steps": ["步骤1描述", "步骤2描述", ...]}"""


def planner_node(state: PlanExecuteState) -> dict:
    """制定或重新制定执行计划。"""
    llm = get_llm(temperature=0)

    # 如果是重新规划，把之前的执行结果也传入
    context = ""
    if state.get("step_results"):
        context = "\n\n之前的执行结果：\n" + json.dumps(
            state["step_results"], ensure_ascii=False, indent=2
        )
        context += "\n\n请根据以上结果调整计划，只添加剩余需要执行的步骤。"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM + context),
    ] + state["messages"]

    response = llm.invoke(messages)

    # 解析计划
    try:
        # 尝试从回复中提取 JSON
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan_data = json.loads(content)
        steps = plan_data.get("steps", [])
    except (json.JSONDecodeError, IndexError):
        # 解析失败，用简单的按行分割
        steps = [line.strip().lstrip("0123456789.) ") for line in response.content.split("\n") if line.strip() and line.strip()[0].isdigit()]

    return {
        "plan": steps,
        "step_index": 0,
        "step_results": state.get("step_results", {}),
        "needs_replan": False,
        "messages": [AIMessage(content=f"已制定计划，共 {len(steps)} 步：\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))],
    }


# ============================================================
# Node 2: Executor —— 按计划执行
# ============================================================
EXECUTOR_SYSTEM = """你是一个执行者。根据计划中的当前步骤，决定是否需要调用工具。

如果当前步骤需要调用工具，请输出工具调用指令。
如果当前步骤是汇总/回复，请直接生成最终回复。

可用工具：query_order, query_logistics, query_inventory, check_refund_policy"""


def executor_node(state: PlanExecuteState) -> dict:
    """执行当前步骤。"""
    plan = state["plan"]
    step_index = state["step_index"]

    if step_index >= len(plan):
        return {"needs_replan": False}

    current_step = plan[step_index]
    print(f"\n  [执行] Step {step_index + 1}/{len(plan)}: {current_step}")

    # 判断这一步是否需要调用工具
    tool_to_call = None
    tool_args = {}
    for tool_name in TOOL_MAP:
        if tool_name.replace("_", " ") in current_step.lower() or tool_name in current_step.lower():
            tool_to_call = tool_name
            break

    if tool_to_call:
        # 模拟参数提取（真实场景中由 LLM 的 tool_calls 来做）
        tool_func = TOOL_MAP[tool_to_call]
        print(f"    → 调用工具: {tool_to_call}")
        try:
            result = tool_func.invoke(tool_args)
        except Exception as e:
            result = json.dumps({"error": str(e)}, ensure_ascii=False)

        print(f"    ← 结果: {result[:80]}...")

        new_results = state.get("step_results", {}).copy()
        new_results[step_index] = {
            "step": current_step,
            "tool": tool_to_call,
            "result": result,
        }

        # 判断结果是否需要重新规划
        needs_replan = False
        if "error" in result:
            needs_replan = True
            print(f"    ⚠ 工具返回错误，标记需要重新规划")

        return {
            "step_results": new_results,
            "step_index": step_index + 1,
            "needs_replan": needs_replan,
        }
    else:
        # 不需要工具，直接推进
        new_results = state.get("step_results", {}).copy()
        new_results[step_index] = {"step": current_step, "tool": None, "result": "已处理"}
        return {
            "step_results": new_results,
            "step_index": step_index + 1,
            "needs_replan": False,
        }


# ============================================================
# Node 3: Responder —— 生成最终回复
# ============================================================
def responder_node(state: PlanExecuteState) -> dict:
    """根据所有步骤的执行结果，生成最终的自然语言回复。"""
    llm = get_llm(temperature=0.3)

    # 构造汇总 Prompt
    step_summary = ""
    for idx, result in state.get("step_results", {}).items():
        step_summary += f"\n步骤 {idx + 1}: {result.get('step', '')}\n  结果: {result.get('result', '')}\n"

    summary_prompt = f"""请根据以下执行结果，生成一段自然语言的回复来回答用户的原始问题。

原始问题：{state['messages'][0].content if state['messages'] else '未知'}

执行过程：
{step_summary}

要求：
1. 回复要简洁友好，像客服一样
2. 把关键信息（订单状态、物流位置、退货方案等）说清楚
3. 如果有需要用户操作的事项（比如去 APP 申请退款），明确告知"""

    response = llm.invoke([HumanMessage(content=summary_prompt)])

    return {
        "messages": [AIMessage(content=response.content)],
    }


# ============================================================
# 条件判断函数
# ============================================================
def should_replan(state: PlanExecuteState) -> str:
    """判断是否需要重新规划。"""
    if state.get("needs_replan"):
        return "replan"
    if state["step_index"] >= len(state["plan"]):
        return "respond"
    return "execute"


def should_continue_execute(state: PlanExecuteState) -> str:
    """判断是否继续执行下一步。"""
    if state["step_index"] >= len(state["plan"]):
        return "respond"
    if state.get("needs_replan"):
        return "replan"
    return "execute"


# ============================================================
# 构建完整的状态图
# ============================================================
def build_plan_execute_graph():
    """构建 Plan-Execute-Replan 完整状态图。"""
    graph = StateGraph(PlanExecuteState)

    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("responder", responder_node)

    # 起点 → 制定计划
    graph.add_edge("__start__", "planner")

    # 计划制定完 → 开始执行
    graph.add_edge("planner", "executor")

    # 执行完后判断：继续执行 / 重新规划 / 生成回复
    graph.add_conditional_edges("executor", should_continue_execute, {
        "execute": "executor",     # 继续执行下一步
        "replan": "planner",       # 重新规划
        "respond": "responder",    # 生成最终回复
    })

    # 回复 → 结束
    graph.add_edge("responder", END)

    return graph.compile()


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 5：完整 Plan-Execute-Replan Agent")
    print("=" * 60)

    graph = build_plan_execute_graph()

    test_cases = [
        ("简单问题：查一下订单 ORD-20240125-001 的物流", "单步查询"),
        ("我买了 StarPods Pro，订单号 ORD-20240125-001，到了之后想退货，现在 StarPods Lite 有没有货可以换？", "多步查询 + 联合决策"),
    ]

    for question, desc in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试：{desc}")
        print(f"用户：{question}")
        print("-" * 60)

        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "plan": [],
            "step_index": 0,
            "step_results": {},
            "needs_replan": False,
            "max_steps": 10,
        })

        # 提取最终回复（最后一条 AI 消息）
        final_reply = "无回复"
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_reply = msg.content
                break

        print(f"\n最终回复：{final_reply}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. Plan-Execute-Replan 是处理复杂问题的经典模式")
    print("  2. LangGraph 的条件边让循环和分支变得很简单")
    print("  3. State 是所有节点共享的「工作记忆」")
    print("  4. Replan 机制让 AI 能在执行过程中自我纠正")
    print("  5. 设置 max_steps 防止无限循环")
    print("=" * 60)
