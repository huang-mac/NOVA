"""
===============================================
第十章 示例 5：MCP + Skills 完整集成
===============================================
目标：把 MCP 协议和 Skills 体系结合起来，实现一个能动态加载技能包、
     自动连接 MCP Server 的智能客服 Agent。

这是本章的核心文件。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


# ============================================================
# 简化 MCP Server
# ============================================================
class SimpleMCPServer:
    """简化的 MCP Server，展示核心协议。"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = {}

    def register_tool(self, name: str, description: str, handler: callable):
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
        }

    def list_tools(self) -> list:
        return [
            {"name": t["name"], "description": t["description"], "server": self.name}
            for t in self.tools.values()
        ]

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        arguments = arguments or {}
        tool = self.tools.get(tool_name)
        if not tool:
            return json.dumps({"error": f"工具 {tool_name} 不存在"}, ensure_ascii=False)
        try:
            result = tool["handler"](**arguments)
            return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# 简化 MCP Client
# ============================================================
class SimpleMCPClient:
    """简化的 MCP Client，连接多个 MCP Server。"""

    def __init__(self):
        self.servers = {}  # name → SimpleMCPServer

    def connect(self, server: SimpleMCPServer):
        self.servers[server.name] = server
        print(f"  [MCP] 已连接 Server: {server.name} ({len(server.tools)} 个工具)")

    def discover_tools(self) -> list:
        """发现所有 Server 上的工具。"""
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.list_tools())
        return all_tools

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """在所有 Server 中查找并调用工具。"""
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.call_tool(tool_name, arguments)
        return json.dumps({"error": f"未找到工具 {tool_name}"}, ensure_ascii=False)


# ============================================================
# Skill 技能包
# ============================================================
class Skill:
    """技能包 —— 一组相关能力的打包。"""

    def __init__(self, name: str, description: str, system_prompt: str,
                 required_servers: list = None, required_tools: list = None,
                 knowledge: str = ""):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.required_servers = required_servers or []   # 需要的 MCP Server 名称
        self.required_tools = required_tools or []       # 需要的工具名
        self.knowledge = knowledge                       # 关联的知识文档


# ============================================================
# SkillManager
# ============================================================
class SkillManager:
    """技能包管理器。"""

    def __init__(self):
        self.skills = {}           # name → Skill
        self.mcp_client = SimpleMCPClient()

    def register_skill(self, skill: Skill):
        self.skills[skill.name] = skill

    def register_server(self, server: SimpleMCPServer):
        self.mcp_client.connect(server)

    def select_skill(self, user_message: str) -> Optional[Skill]:
        """根据用户消息选择最合适的技能包。"""
        llm = get_llm(temperature=0)

        skill_list = "\n".join(
            f"- {name}: {skill.description}"
            for name, skill in self.skills.items()
        )

        prompt = f"""根据用户消息，选择最合适的技能包。

可用技能包：
{skill_list}

请只输出技能包名称，不要输出其他内容。如果不确定，选最接近的。"""

        response = llm.invoke([HumanMessage(content=prompt)])
        skill_name = response.content.strip().lower()

        # 模糊匹配
        for name in self.skills:
            if name in skill_name or skill_name in name:
                return self.skills[name]

        # 默认返回第一个
        return list(self.skills.values())[0] if self.skills else None

    def get_skill_tools(self, skill: Skill) -> list:
        """获取技能包可用的工具列表。"""
        available = self.mcp_client.discover_tools()
        # 过滤出该技能包需要的工具
        skill_tool_names = set(skill.required_tools)
        return [t for t in available if t["name"] in skill_tool_names or not skill_tool_names]


# ============================================================
# 完整的 MCP + Skills Agent
# ============================================================
class MCPSkillsAgent:
    """结合 MCP 和 Skills 的智能客服 Agent。"""

    def __init__(self, skill_manager: SkillManager):
        self.skill_manager = skill_manager
        self.llm = get_llm(temperature=0)

    def chat(self, user_message: str) -> dict:
        """处理用户消息。"""
        print(f"\n  [Agent] 收到消息：{user_message}")

        # Step 1: 选择技能包
        print("  [Agent] 正在选择技能包...")
        skill = self.skill_manager.select_skill(user_message)
        print(f"  [Agent] 选择技能包：{skill.name} - {skill.description}")

        # Step 2: 获取可用工具
        available_tools = self.skill_manager.get_skill_tools(skill)
        print(f"  [Agent] 可用工具：{[t['name'] for t in available_tools]}")

        # Step 3: 构造带技能包的 Prompt
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in available_tools
        )

        system_prompt = skill.system_prompt
        if tool_descriptions:
            system_prompt += f"\n\n可用工具：\n{tool_descriptions}\n\n工具调用规则：需要时通过工具名调用，参数用 JSON 格式。"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]

        # Step 4: 处理（支持多轮工具调用）
        tool_calls_log = []
        for _ in range(3):
            response = self.llm.invoke(messages)
            messages.append(response)

            if not hasattr(response, 'tool_calls') or not response.tool_calls:
                # 尝试从文本中解析工具调用意图（简化版）
                content = response.content
                if "调用工具" in content or "call" in content.lower():
                    # 简化处理：用 MCP Client 直接调用
                    for tool_info in available_tools:
                        if tool_info["name"] in content.lower():
                            result = self.skill_manager.mcp_client.call_tool(tool_info["name"])
                            tool_calls_log.append({"tool": tool_info["name"], "result": result[:80]})
                            messages.append(ToolMessage(content=result, tool_call_id="manual"))
                            break
                    else:
                        break
                else:
                    break
            else:
                # 标准 tool_calls 处理
                for tc in response.tool_calls:
                    result = self.skill_manager.mcp_client.call_tool(tc["name"], tc.get("args"))
                    tool_calls_log.append({"tool": tc["name"], "result": result[:80]})
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        final_reply = response.content if hasattr(response, 'content') and response.content else "处理完成"

        return {
            "reply": final_reply,
            "skill": skill.name,
            "tools_available": [t["name"] for t in available_tools],
            "tools_called": [tc["tool"] for tc in tool_calls_log],
        }


# ============================================================
# 初始化 MCP Server 和 Skills
# ============================================================
def create_system():
    """创建完整的 MCP + Skills 系统。"""

    # 创建 MCP Server
    order_server = SimpleMCPServer("order_service", "订单管理服务")
    order_server.register_tool("query_order", "查询订单详情",
        lambda order_id: MOCK_ORDERS.get(order_id, {"error": "未找到"}))
    order_server.register_tool("query_logistics", "查询物流信息",
        lambda order_id: MOCK_LOGISTICS.get(order_id, {"error": "暂无物流"}))

    product_server = SimpleMCPServer("product_service", "产品信息服务")
    product_server.register_tool("query_inventory", "查询商品库存",
        lambda product_name: MOCK_INVENTORY.get(product_name, {"error": "未找到"}))
    product_server.register_tool("check_refund_policy", "查询退款政策",
        lambda product_name, order_days=0: {"policy": "7天退货", "warranty": "1年质保",
                                               "note": "在退货期内" if order_days <= 7 else "超出退货期，在质保期内"})

    # 创建 Skill
    refund_skill = Skill(
        name="refund",
        description="处理退款、退货、换货请求",
        system_prompt="""你是星辰科技的退款专员。
规则：查订单 → 判断退货条件 → 告知用户流程。
语气：亲切、专业、简洁。""",
        required_servers=["order_service", "product_service"],
        required_tools=["query_order", "check_refund_policy"],
    )

    logistics_skill = Skill(
        name="logistics",
        description="查询物流和订单配送状态",
        system_prompt="""你是星辰科技的物流查询专员。
规则：查订单 → 查物流 → 告知用户当前位置和预计到达时间。
语气：简洁、高效。""",
        required_servers=["order_service"],
        required_tools=["query_order", "query_logistics"],
    )

    tech_skill = Skill(
        name="tech_support",
        description="处理产品故障和使用咨询",
        system_prompt="""你是星辰科技的技术支持工程师。
规则：了解症状 → 给出排查建议 → 建议维修或换新。
语气：耐心、专业。""",
        required_servers=["order_service", "product_service"],
        required_tools=["query_order", "query_inventory"],
    )

    # 组装系统
    manager = SkillManager()
    manager.register_server(order_server)
    manager.register_server(product_server)
    manager.register_skill(refund_skill)
    manager.register_skill(logistics_skill)
    manager.register_skill(tech_skill)

    return MCPSkillsAgent(manager)


# 模拟数据
MOCK_ORDERS = {"ORD-20240125-001": {"product": "StarPods Pro", "status": "shipped", "days": 10}}
MOCK_LOGISTICS = {"ORD-20240125-001": {"courier": "顺丰", "location": "北京", "eta": "明天"}}
MOCK_INVENTORY = {"StarPods Pro": {"stock": 156, "status": "有货"}}


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第十章 示例 5：MCP + Skills 完整集成")
    print("=" * 60)

    agent = create_system()

    test_cases = [
        ("帮我查一下订单 ORD-20240125-001 的物流", "物流查询"),
        ("我的耳机买了 10 天了能退吗？", "退款咨询"),
        ("StarPods Pro 还有没有货？", "库存查询"),
    ]

    for question, desc in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试：{desc}")
        print(f"用户：{question}")
        print("-" * 60)

        result = agent.chat(question)

        print(f"\n  [结果] 使用技能包：{result['skill']}")
        print(f"  [结果] 可用工具：{result['tools_available']}")
        print(f"  [结果] 调用工具：{result['tools_called']}")
        print(f"  [结果] 回复：{result['reply']}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. MCP 标准化了工具接入方式，新增工具不需要改 Agent 代码")
    print("  2. Skills 把能力模块化，每个技能包有自己的 Prompt + 工具 + 知识")
    print("  3. SkillManager 根据用户问题自动选择技能包")
    print("  4. 技能包声明需要的 MCP Server，加载时自动连接")
    print("  5. 新增业务能力 = 新建 Skill + MCP Server，主系统零改动")
    print("=" * 60)
