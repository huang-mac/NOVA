"""
===============================================
第九章 示例 2：专业 Agent —— 各司其职
===============================================
目标：实现售后、技术支持、主管三个专业 Agent，
     每个 Agent 有独立的 Prompt、工具和职责。

每个专业 Agent 就是一个带着专属工具和 Prompt 的"员工"。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from config import get_llm


# ============================================================
# 模拟数据
# ============================================================
MOCK_ORDERS = {
    "ORD-20240125-001": {"order_id": "ORD-20240125-001", "product": "StarPods Pro", "status": "shipped", "days": 10},
    "ORD-20240128-002": {"order_id": "ORD-20240128-002", "product": "StarPods Lite", "status": "delivered", "days": 3},
}
MOCK_LOGISTICS = {"ORD-20240125-001": {"courier": "顺丰", "location": "北京分拨中心", "eta": "明天"}}
MOCK_INVENTORY = {"StarPods Pro": {"stock": 156, "status": "有货"}, "StarPods Lite": {"stock": 0, "status": "缺货"}}


# ============================================================
# 工具
# ============================================================
@tool
def query_order(order_id: str) -> str:
    """查询订单详情。"""
    return json.dumps(MOCK_ORDERS.get(order_id, {"error": "未找到"}), ensure_ascii=False)

@tool
def query_logistics(order_id: str) -> str:
    """查询物流。"""
    return json.dumps(MOCK_LOGISTICS.get(order_id, {"error": "暂无物流"}), ensure_ascii=False)

@tool
def check_refund_policy(product: str, days: int = 0) -> str:
    """查询退款政策。"""
    note = "在7天退货期内" if days <= 7 else f"已超7天(购于{days}天前)，但在1年质保期内"
    return json.dumps({"policy": note, "return_period": "7天", "warranty": "1年"}, ensure_ascii=False)

@tool
def query_inventory(product_name: str) -> str:
    """查询库存。"""
    for name, item in MOCK_INVENTORY.items():
        if product_name.lower() in name.lower():
            return json.dumps(item, ensure_ascii=False)
    return json.dumps({"error": f"未找到 {product_name}"}, ensure_ascii=False)


# ============================================================
# Agent 基类
# ============================================================
class BaseAgent:
    """
    Agent 基类 —— 提供统一的消息处理和工具调用能力。

    所有专业 Agent 继承这个基类，只需定义自己的 name、role、
    system_prompt 和 tools 即可。
    """

    def __init__(self, name: str, role: str, system_prompt: str, tools: list = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.llm = get_llm(temperature=0)
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    def process(self, message: str, context: str = "") -> dict:
        """
        处理消息，返回回复和元数据。

        Args:
            message: 用户消息
            context: 来自上游的上下文信息（如接待 Agent 提取的关键信息）

        Returns:
            {"agent": name, "reply": str, "tool_calls": list}
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"{context}\n\n用户消息：{message}" if context else message),
        ]

        tool_calls_log = []

        # 最多 3 轮工具调用
        for _ in range(3):
            response = self.llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                tool_func = None
                for t in self.tools:
                    if t.name == tool_name:
                        tool_func = t
                        break

                result = tool_func.invoke(tool_args) if tool_func else json.dumps({"error": f"未知工具 {tool_name}"}, ensure_ascii=False)
                tool_calls_log.append({"tool": tool_name, "args": tool_args, "result": result[:80]})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return {
            "agent": self.name,
            "reply": response.content if hasattr(response, 'content') and response.content else "处理完成",
            "tool_calls": tool_calls_log,
        }


# ============================================================
# 售后专员 Agent
# ============================================================
class AfterSalesAgent(BaseAgent):
    """售后专员 —— 处理退款、退货、换货、物流查询。"""

    def __init__(self):
        super().__init__(
            name="after_sales",
            role="售后专员",
            system_prompt="""你是星辰科技的售后专员，负责处理退款、退货、换货、物流查询。

规则：
1. 查到结果后用自然语言总结，不要返回原始数据
2. 退款/退货引导用户在 APP 中申请
3. 物流问题告知当前位置和预计到达时间
4. 超出退货期但在质保期内的情况要主动说明
5. 语气亲切专业，简洁不废话""",
            tools=[query_order, query_logistics, check_refund_policy],
        )


# ============================================================
# 技术支持 Agent
# ============================================================
class TechSupportAgent(BaseAgent):
    """技术支持工程师 —— 处理产品故障和使用咨询。"""

    def __init__(self):
        super().__init__(
            name="tech_support",
            role="技术支持工程师",
            system_prompt="""你是星辰科技的技术支持工程师，负责处理产品故障和使用咨询。

规则：
1. 先了解具体症状和产品型号
2. 给出逐步排查建议
3. 如果是硬件故障，建议用户申请售后维修
4. 语气专业耐心，避免使用过于技术化的术语""",
            tools=[query_order, query_inventory],
        )


# ============================================================
# 主管 Agent
# ============================================================
class SupervisorAgent(BaseAgent):
    """客服主管 —— 处理投诉、审核工单和兜底。"""

    def __init__(self):
        super().__init__(
            name="supervisor",
            role="客服主管",
            system_prompt="""你是星辰科技的客服主管，负责处理投诉、审核工单和兜底。

规则：
1. 投诉用户要先安抚情绪，表示重视
2. 了解问题全貌后再给出方案
3. 可以给出补偿方案（如优惠券、延长质保等）
4. 复杂问题及时创建工单并分配跟进
5. 语气诚恳，展现解决问题的诚意""",
            tools=[query_order, query_logistics, check_refund_policy, query_inventory],
        )


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第九章 示例 2：专业 Agent —— 各司其职")
    print("=" * 60)

    agents = {
        "售后": AfterSalesAgent(),
        "技术支持": TechSupportAgent(),
        "主管": SupervisorAgent(),
    }

    test_cases = [
        ("售后", "我要退订单 ORD-20240125-001，买了 10 天了"),
        ("技术支持", "我的 StarPods Pro 有杂音怎么办？"),
        ("主管", "你们耳机质量太差了，用了两天就坏了，我要投诉！"),
    ]

    for role, question in test_cases:
        print(f"\n{'-' * 60}")
        print(f"[{role} Agent] 用户：{question}")

        agent = agents[role]
        result = agent.process(question)

        print(f"  回复：{result['reply']}")
        if result["tool_calls"]:
            print(f"  调用工具：{[tc['tool'] for tc in result['tool_calls']]}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. 每个 Agent 有独立的 Prompt，专注自己的领域")
    print("  2. 每个 Agent 有不同的工具集，按需分配")
    print("  3. BaseAgent 基类封装了通用的消息处理和工具调用逻辑")
    print("  4. 专业 Agent 可以接收上游上下文（如接待 Agent 提取的关键信息）")
    print("  5. 主管 Agent 拥有所有工具权限，用于兜底处理")
    print("=" * 60)
