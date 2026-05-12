"""
===============================================
第七章 示例 1：Tool 基础概念
===============================================
目标：理解 LangChain Tool 的定义方式、绑定方式，以及 Function Call 的完整生命周期。

这一章是"让 AI 从只会说话到能办事"的第一步。
先从一个最简单的乘法工具开始，搞清楚 Function Call 是怎么跑的。
"""

import sys
import os

# 把共享配置加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 定义工具
# ============================================================
# @tool 装饰器把一个普通 Python 函数变成 LLM 可以调用的 Tool。
# LLM 会根据函数名 + docstring + 参数类型注解来判断什么时候该用它。

@tool
def multiply(a: int, b: int) -> int:
    """两个数字相乘。当用户需要进行乘法计算时使用这个工具。

    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """两个数字相加。当用户需要进行加法计算时使用这个工具。

    Args:
        b: 第二个数字
    """
    return a + b


# ============================================================
# 2. 查看工具的 schema
# ============================================================
# Tool 会自动生成一个 JSON Schema，这就是 LLM "看到"的工具说明书。

def show_tool_schema():
    """打印工具的 JSON Schema，看看 LLM 实际看到的是什么。"""
    print("=== multiply 工具的 Schema ===")
    print(multiply.name)        # 工具名
    print(multiply.description) # 工具描述（来自 docstring）
    print(multiply.args_schema.model_json_schema())  # 参数结构
    print()


# ============================================================
# 3. 绑定工具到 LLM
# ============================================================
# llm.bind_tools() 把工具列表"告诉"LLM。
# 之后 LLM 在生成回复时，如果判断需要调用工具，就会输出 tool_calls 而不是普通文字。

def basic_tool_call():
    """演示一次完整的 Function Call 流程。"""
    from config import get_llm

    llm = get_llm(temperature=0)

    # 把工具绑到 LLM 上
    llm_with_tools = llm.bind_tools([multiply, add])

    # 模拟用户提问
    question = "帮我算一下 23 乘以 47 是多少？"
    print(f"用户问题：{question}")

    # 第一次调用 LLM —— 它会判断需要调用 multiply 工具
    print("\n--- 第一次 LLM 调用 ---")
    response = llm_with_tools.invoke(question)

    # 检查 LLM 是否决定调用工具
    if response.tool_calls:
        print(f"LLM 决定调用 {len(response.tool_calls)} 个工具：")
        for tc in response.tool_calls:
            print(f"  - 工具名: {tc['name']}")
            print(f"  - 参数: {tc['args']}")

        # 手动执行工具调用
        print("\n--- 执行工具调用 ---")
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            # 找到对应的工具并执行
            if tool_name == "multiply":
                result = multiply.invoke(tool_args)
            elif tool_name == "add":
                result = add.invoke(tool_args)
            else:
                result = f"未知工具: {tool_name}"

            print(f"  {tool_name}({tool_args}) = {result}")

        # 把工具执行结果喂回给 LLM，让它生成自然语言回复
        print("\n--- 第二次 LLM 调用（生成最终回复）---")

        # 构造 ToolMessage（告诉 LLM 工具执行的结果）
        from langchain_core.messages import ToolMessage
        tool_messages = []
        for tc in response.tool_calls:
            if tc["name"] == "multiply":
                result = multiply.invoke(tc["args"])
            elif tc["name"] == "add":
                result = add.invoke(tc["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

        # LLM 根据工具结果生成最终回复
        final_response = llm_with_tools.invoke(
            [response] + tool_messages
        )
        print(f"最终回复：{final_response.content}")
    else:
        # LLM 认为不需要调用工具，直接回复
        print(f"LLM 直接回复：{response.content}")


# ============================================================
# 4. LLM 自己决定不用工具的场景
# ============================================================
def no_tool_needed():
    """有时候 LLM 会判断不需要调用工具。"""
    from config import get_llm

    llm = get_llm(temperature=0)
    llm_with_tools = llm.bind_tools([multiply, add])

    question = "你好，今天天气怎么样？"
    print(f"\n用户问题：{question}")
    print("--- LLM 调用 ---")

    response = llm_with_tools.invoke(question)
    if response.tool_calls:
        print(f"LLM 调用了工具（不应该）：{response.tool_calls}")
    else:
        print(f"LLM 判断不需要工具，直接回复：{response.content}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第七章 示例 1：Tool 基础概念")
    print("=" * 60)

    # 先看看工具长什么样
    show_tool_schema()

    # 演示完整的 Function Call 流程
    basic_tool_call()

    # 演示 LLM 判断不需要工具的场景
    no_tool_needed()

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. @tool 装饰器把 Python 函数变成 LLM 可以调用的工具")
    print("  2. docstring 就是工具的'说明书'，LLM 靠它判断什么时候该用")
    print("  3. llm.bind_tools() 把工具列表告诉 LLM")
    print("  4. Function Call 是两步的：LLM 决定调 → 代码执行 → 结果喂回 LLM")
    print("  5. LLM 有时会判断不需要工具，这时直接回复")
    print("=" * 60)
