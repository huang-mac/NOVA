"""
===============================================
第九章 示例 4：完整 Multi-Agent 客服系统
===============================================
目标：实现一个多智能体协作的客服系统——接待 Agent 负责分流，
     售后 / 技术支持 / 主管 Agent 各司其职。

这是本章的核心文件，展示了 Multi-Agent 在客服场景中的完整实现。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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


# ============================================================
# Agent 定义
# ============================================================
class BaseAgent:
    """Agent 基类。"""
    def __init__(self, name: str, role: str, system_prompt: str, tools: list = None):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.llm = get_llm(temperature=0)
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    def process(self, message: str, context: str = "") -> dict:
        """处理消息，返回回复和元数据。"""
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

                # 查找工具
                tool_func = None
                for t in self.tools:
                    if t.name == tool_name:
                        tool_func = t
                        break

                result = tool_func.invoke(tool_args) if tool_func else json.dumps({"error": f"未知工具 {tool_name}"}, ensure_ascii=False)

                tool_calls_log.append({"tool": tool_name, "args": tool_args, "result": result[:80]})

                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return {
            "agent": self.name,
            "reply": response.content if hasattr(response, 'content') and response.content else "处理完成",
            "tool_calls": tool_calls_log,
        }


# ============================================================
# 接待 Agent（路由）
# ============================================================
class ReceptionAgent(BaseAgent):
    """前台接待 Agent —— 分析意图，分配给专业 Agent。"""
    ROUTE_MAP = {
        "after_sales": "售后专员",
        "tech_support": "技术支持",
        "supervisor": "客服主管",
    }

    def __init__(self):
        super().__init__(
            name="reception",
            role="前台接待",
            system_prompt="""你是星辰科技客服的前台接待。你的唯一工作是分析用户消息，判断该派给哪个专员处理。

可选路由：
1. after_sales（售后专员）—— 退款、退货、换货、物流查询、订单状态
2. tech_support（技术支持）—— 产品故障、使用方法、连接问题、功能咨询
3. supervisor（客服主管）—— 投诉、对之前处理不满、复杂问题、说不清的问题

请用 JSON 格式输出：
{"route": "after_sales/tech_support/supervisor", "reason": "简要原因", "extracted_info": {"order_id": "...", "product": "...", "key_info": "..."}}""",
        )

    def route(self, message: str) -> dict:
        """路由决策。"""
        result = super().process(message)
        try:
            content = result["reply"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            decision = json.loads(content)
            return {
                "agent": decision.get("route", "supervisor"),
                "reason": decision.get("reason", ""),
                "extracted_info": decision.get("extracted_info", {}),
            }
        except (json.JSONDecodeError, IndexError):
            return {"agent": "supervisor", "reason": "无法识别意图，转主管处理", "extracted_info": {}}


# ============================================================
# 专业 Agent
# ============================================================
class AfterSalesAgent(BaseAgent):
    """售后专员 Agent。"""
    def __init__(self):
        super().__init__(
            name="after_sales",
            role="售后专员",
            system_prompt="""你是星辰科技的售后专员，负责处理退款、退货、换货、物流查询。

规则：
1. 查到结果后用自然语言总结，不要返回原始数据
2. 退款/退货引导用户在 APP 中申请
3. 物流问题告知当前位置和预计到达时间
4. 超出退货期但 在质保期内的情况要主动说明
5. 语气亲切专业，简洁不废话""",
            tools=[query_order, query_logistics, check_refund_policy],
        )


class TechSupportAgent(BaseAgent):
    """技术支持 Agent。"""
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
            tools=[query_order],
        )


class SupervisorAgent(BaseAgent):
    """客服主管 Agent。"""
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
            tools=[query_order, query_logistics, check_refund_policy],
        )


# ============================================================
# Multi-Agent 系统调度器
# ============================================================
class MultiAgentSystem:
    """多智能体客服系统调度器。"""

    def __init__(self):
        self.reception = ReceptionAgent()
        self.agents = {
            "after_sales": AfterSalesAgent(),
            "tech_support": TechSupportAgent(),
            "supervisor": SupervisorAgent(),
        }

    def chat(self, user_message: str) -> dict:
        """
        处理用户消息的完整流程：
        1. 接待 Agent 分析意图
        2. 分配给对应的专业 Agent
        3. 专业 Agent 处理并返回结果
        """
        print(f"\n  [系统] 用户消息：{user_message}")

        # Step 1: 接待 Agent 路由
        print(f"  [接待] 正在分析意图...")
        route_decision = self.reception.route(user_message)
        target_agent = route_decision["agent"]
        reason = route_decision["reason"]
        extracted_info = route_decision.get("extracted_info", {})

        print(f"  [接待] 路由决策：{self.reception.ROUTE_MAP.get(target_agent, target_agent)}")
        print(f"  [接待] 原因：{reason}")

        # Step 2: 专业 Agent 处理
        agent = self.agents.get(target_agent, self.agents["supervisor"])
        print(f"  [{agent.role}] 正在处理...")

        # 把提取的信息作为上下文传递
        context = ""
        if extracted_info:
            context = f"接待 Agent 提取的信息：{json.dumps(extracted_info, ensure_ascii=False)}"

        result = agent.process(user_message, context=context)

        # 汇总
        return {
            "reply": result["reply"],
            "routed_to": agent.role,
            "route_reason": reason,
            "tool_calls": result.get("tool_calls", []),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第九章 示例 4：完整 Multi-Agent 客服系统")
    print("=" * 60)

    system = MultiAgentSystem()

    test_cases = [
        ("我的耳机有杂音怎么办？", "技术问题"),
        ("我要查一下订单 ORD-20240125-001 到哪了", "物流查询"),
        ("你们这个耳机质量太差了！用了两天就坏了，我要投诉！", "投诉"),
        ("StarPods Pro 买了 10 天了能退吗？", "退货咨询"),
    ]

    for question, desc in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试：{desc}")
        print(f"用户：{question}")
        print("-" * 60)

        result = system.chat(question)

        print(f"\n  [结果] 分配给：{result['routed_to']}")
        print(f"  [结果] 回复：{result['reply']}")
        if result["tool_calls"]:
            print(f"  [结果] 调用工具：{[tc['tool'] for tc in result['tool_calls']]}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. 每个 Agent 有独立的 Prompt、工具和职责")
    print("  2. 接待 Agent 做路由，不处理具体业务")
    print("  3. 上下文（提取的订单号、产品名等）会传递给专业 Agent")
    print("  4. Multi-Agent 比单 Agent 更专业、更好维护")
    print("=" * 60)
