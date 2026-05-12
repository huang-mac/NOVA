"""
===============================================
第九章 示例 1：接待 Agent —— 前台分流
===============================================
目标：实现一个前台接待 Agent，负责接收用户消息、识别意图、
     分配给对应的专业 Agent 处理。

接待 Agent 不需要调工具，核心是一个 LLM 驱动的意图分类器。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


# ============================================================
# 接待 Agent 的路由定义
# ============================================================
ROUTE_OPTIONS = {
    "after_sales": "售后专员",
    "tech_support": "技术支持",
    "supervisor": "客服主管",
}


# ============================================================
# 接待 Agent —— 意图识别 + 路由
# ============================================================
class ReceptionAgent:
    """
    前台接待 Agent —— 分析用户意图，决定分派给哪个专员。

    这是 Multi-Agent 系统的"入口"，类似于公司的前台：
    不处理具体业务，只负责听清用户需求、分派给对的人。
    """

    ROUTE_PROMPT = """你是星辰科技客服的前台接待。你的唯一工作是分析用户消息，判断该派给哪个专员处理。

可选路由：
1. after_sales（售后专员）—— 退款、退货、换货、物流查询、订单状态
2. tech_support（技术支持）—— 产品故障、使用方法、连接问题、功能咨询
3. supervisor（客服主管）—— 投诉、对之前处理不满、复杂问题、说不清的问题

判断依据：
- 提到"退款""退货""换货""物流""到哪了""发货""签收" → after_sales
- 提到"坏了""连不上""杂音""怎么用""设置""故障" → tech_support
- 提到"投诉""不满""你们什么态度""经理""太差了" → supervisor
- 无法判断 → supervisor（兜底）

同时提取关键信息：订单号、产品名、问题描述。

请用 JSON 格式输出：
{"route": "after_sales/tech_support/supervisor", "reason": "简要原因", "extracted_info": {"order_id": "...", "product": "...", "key_info": "..."}}"""

    def __init__(self):
        self.llm = get_llm(temperature=0)

    def route(self, message: str) -> dict:
        """
        分析用户消息，返回路由决策。

        Args:
            message: 用户的消息

        Returns:
            路由决策字典，如：
            {
                "agent": "after_sales",
                "reason": "用户询问退款流程",
                "extracted_info": {"order_id": "ORD-001", "product": "StarPods Pro"}
            }
        """
        messages = [
            SystemMessage(content=self.ROUTE_PROMPT),
            HumanMessage(content=message),
        ]

        response = self.llm.invoke(messages)

        # 解析路由决策
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            decision = json.loads(content)
            route = decision.get("route", "supervisor")

            # 校验路由是否合法
            if route not in ROUTE_OPTIONS:
                route = "supervisor"

            return {
                "agent": route,
                "agent_name": ROUTE_OPTIONS[route],
                "reason": decision.get("reason", ""),
                "extracted_info": decision.get("extracted_info", {}),
            }
        except (json.JSONDecodeError, IndexError):
            return {
                "agent": "supervisor",
                "agent_name": "客服主管",
                "reason": "无法识别意图，转主管处理",
                "extracted_info": {},
            }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第九章 示例 1：接待 Agent —— 前台分流")
    print("=" * 60)

    agent = ReceptionAgent()

    test_cases = [
        "我的耳机有杂音怎么办？",
        "我要查一下订单 ORD-20240125-001 到哪了",
        "你们这个耳机质量太差了！用了两天就坏了，我要投诉！",
        "StarPods Pro 买了 10 天了能退吗？",
        "你好",  # 模糊意图，应该兜底到 supervisor
    ]

    for question in test_cases:
        print(f"\n{'-' * 60}")
        print(f"用户：{question}")

        result = agent.route(question)

        print(f"  → 分配给：{result['agent_name']}")
        print(f"  → 原因：{result['reason']}")
        if result["extracted_info"]:
            print(f"  → 提取信息：{json.dumps(result['extracted_info'], ensure_ascii=False)}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. 接待 Agent 是 Multi-Agent 系统的入口，只做路由不做业务")
    print("  2. 用 LLM 的结构化输出（JSON）实现意图分类")
    print("  3. 路由时要同时提取关键信息（订单号、产品名等）传递给下游")
    print("  4. 兜底策略很重要：无法识别时转给主管，而不是返回错误")
    print("  5. 校验路由合法性，防止 LLM 输出不在选项中的路由")
    print("=" * 60)
