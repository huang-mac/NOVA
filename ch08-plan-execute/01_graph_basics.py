"""
===============================================
第八章 示例 1：LangGraph 状态机基础
===============================================
目标：理解 LangGraph 的核心概念——State、Node、Edge。

LangGraph 是 LangChain 生态里用来构建"有状态的 AI 工作流"的框架。
如果说 LangChain 的 Chain 是一条直线（A → B → C），那 LangGraph 就是一个有分叉、
有循环的网络图，可以让 AI 的行为更灵活、更智能。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ============================================================
# 1. 定义 State —— 工作流的"共享记忆"
# ============================================================
# State 是整个工作流中所有节点共享的数据结构。
# 每个节点可以读取 State，也可以修改 State。

class SimpleState(TypedDict):
    """最简单的状态结构。"""
    input: str           # 用户输入
    processed: str       # 处理结果
    count: int           # 计数器（演示状态修改）


class ChatState(TypedDict):
    """对话状态结构，用 add_messages 聚合消息列表。"""
    messages: Annotated[list, add_messages]  # 对话消息列表（自动追加）
    step_count: int                           # 当前执行到第几步


# ============================================================
# 2. 定义 Node —— 状态机里的"处理节点"
# ============================================================
# 每个 Node 是一个 Python 函数，接收当前 State，返回 State 的更新。

def step1_process_input(state: SimpleState) -> dict:
    """第一步：处理用户输入。"""
    text = state["input"].strip().upper()
    return {"processed": text, "count": state.get("count", 0) + 1}


def step2_format_output(state: SimpleState) -> dict:
    """第二步：格式化输出。"""
    processed = state["processed"]
    formatted = f"[{state['count']}] 处理结果: {processed}"
    return {"processed": formatted}


# ============================================================
# 3. 构建状态图 —— 把节点连起来
# ============================================================
def build_simple_graph():
    """构建一个最简单的三步走状态机。"""
    # 创建图
    graph = StateGraph(SimpleState)

    # 添加节点
    graph.add_node("process", step1_process_input)
    graph.add_node("format", step2_format_output)

    # 添加边（定义节点之间的执行顺序）
    graph.add_edge("__start__", "process")  # 起点 → process
    graph.add_edge("process", "format")     # process → format
    graph.add_edge("format", END)           # format → 结束

    return graph.compile()


# ============================================================
# 4. 运行状态机
# ============================================================
def run_simple_graph():
    """运行简单的状态机。"""
    print("=== 最简单的状态机 ===")

    graph = build_simple_graph()

    # 输入
    initial_state = {"input": "hello langgraph", "count": 0}
    print(f"输入: {initial_state}")

    # 运行
    result = graph.invoke(initial_state)
    print(f"输出: {result}")
    print(f"  - processed: {result['processed']}")
    print(f"  - count: {result['count']}")


# ============================================================
# 5. 带条件分支的状态机
# ============================================================
def route_by_length(state: SimpleState) -> str:
    """根据输入长度决定走哪条路。"""
    if len(state["input"]) > 10:
        return "long_text"
    return "short_text"


def handle_long_text(state: SimpleState) -> dict:
    return {"processed": f"[长文本处理] {state['input'][:20]}..."}


def handle_short_text(state: SimpleState) -> dict:
    return {"processed": f"[短文本处理] {state['input']}"}


def build_conditional_graph():
    """构建带条件分支的状态机。"""
    graph = StateGraph(SimpleState)

    graph.add_node("process", step1_process_input)
    graph.add_node("long_handler", handle_long_text)
    graph.add_node("short_handler", handle_short_text)

    graph.add_edge("__start__", "process")
    # process 节点执行完后，根据条件走不同的路
    graph.add_conditional_edges("process", route_by_length, {
        "long_text": "long_handler",
        "short_text": "short_handler",
    })
    graph.add_edge("long_handler", END)
    graph.add_edge("short_handler", END)

    return graph.compile()


def run_conditional_graph():
    """运行带条件分支的状态机。"""
    print("\n=== 带条件分支的状态机 ===")

    graph = build_conditional_graph()

    # 测试长文本
    result1 = graph.invoke({"input": "这是一段比较长的文本内容", "count": 0})
    print(f"长文本输入 → {result1['processed']}")

    # 测试短文本
    result2 = graph.invoke({"input": "短文本", "count": 0})
    print(f"短文本输入 → {result2['processed']}")


# ============================================================
# 6. 带循环的状态机 —— Plan-Execute 的雏形
# ============================================================
class LoopState(TypedDict):
    """带循环的状态。"""
    target: int        # 目标数字
    current: int       # 当前数字
    steps: list        # 执行记录


def increment(state: LoopState) -> dict:
    """每次加 1。"""
    new_current = state["current"] + 1
    new_steps = state["steps"] + [f"current = {new_current}"]
    return {"current": new_current, "steps": new_steps}


def should_continue(state: LoopState) -> str:
    """判断是否继续循环。"""
    if state["current"] < state["target"]:
        return "continue"
    return "done"


def build_loop_graph():
    """构建带循环的状态机。"""
    graph = StateGraph(LoopState)

    graph.add_node("increment", increment)

    graph.add_edge("__start__", "increment")
    # increment 执行完后，根据条件决定继续还是结束
    graph.add_conditional_edges("increment", should_continue, {
        "continue": "increment",  # 继续循环
        "done": END,              # 结束
    })

    return graph.compile()


def run_loop_graph():
    """运行带循环的状态机。"""
    print("\n=== 带循环的状态机（Plan-Execute 雏形）===")

    graph = build_loop_graph()
    result = graph.invoke({"target": 5, "current": 0, "steps": []})
    print(f"目标: {result['target']}")
    print(f"最终值: {result['current']}")
    print(f"执行记录: {result['steps']}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第八章 示例 1：LangGraph 状态机基础")
    print("=" * 60)

    # 最简单的状态机
    run_simple_graph()

    # 带条件分支的状态机
    run_conditional_graph()

    # 带循环的状态机
    run_loop_graph()

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. State 是工作流的共享数据，所有节点都能读写")
    print("  2. Node 是处理节点，接收 State、返回 State 更新")
    print("  3. Edge 定义节点间的执行顺序（固定边 + 条件边）")
    print("  4. 条件边可以让工作流根据中间结果走不同路线")
    print("  5. 循环边（回到之前的节点）是 Plan-Execute 的基础")
    print("=" * 60)
