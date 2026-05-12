"""
===============================================
第十章 示例 4：SkillManager 技能包管理器
===============================================
目标：实现技能包管理器，根据用户问题自动选择最合适的技能包，
     并连接对应的 MCP Server 获取工具。

SkillManager 是 MCP + Skills 体系的"大脑"——
它知道有哪些技能包、每个技能包需要什么工具，能根据用户问题自动匹配。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional
from config import get_llm
from langchain_core.messages import HumanMessage


# ============================================================
# 简化 MCP Server / Client（内联）
# ============================================================
class SimpleMCPServer:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = {}

    def register_tool(self, name: str, description: str, handler: callable):
        self.tools[name] = {"name": name, "description": description, "handler": handler}

    def list_tools(self) -> list:
        return [{"name": t["name"], "description": t["description"], "server": self.name}
                for t in self.tools.values()]

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


class SimpleMCPClient:
    def __init__(self):
        self.servers = {}

    def connect(self, server: SimpleMCPServer):
        self.servers[server.name] = server

    def discover_tools(self) -> list:
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.list_tools())
        return all_tools

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        for server in self.servers.values():
            if tool_name in server.tools:
                return server.call_tool(tool_name, arguments)
        return json.dumps({"error": f"未找到工具 {tool_name}"}, ensure_ascii=False)


# ============================================================
# Skill
# ============================================================
class Skill:
    def __init__(self, name: str, description: str, system_prompt: str = "",
                 required_servers: list = None, required_tools: list = None,
                 knowledge: str = ""):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.required_servers = required_servers or []
        self.required_tools = required_tools or []
        self.knowledge = knowledge


# ============================================================
# SkillManager
# ============================================================
class SkillManager:
    """
    技能包管理器。

    职责：
    1. 管理所有注册的技能包
    2. 管理已连接的 MCP Server
    3. 根据用户问题选择最合适的技能包
    4. 为选中的技能包获取可用工具列表
    """

    def __init__(self):
        self.skills = {}           # name → Skill
        self.mcp_client = SimpleMCPClient()

    def register_skill(self, skill: Skill):
        """注册一个技能包。"""
        self.skills[skill.name] = skill
        print(f"  [SkillManager] 注册技能包: {skill.name} - {skill.description}")

    def register_server(self, server: SimpleMCPServer):
        """注册一个 MCP Server。"""
        self.mcp_client.connect(server)
        print(f"  [SkillManager] 连接 Server: {server.name} ({len(server.tools)} 个工具)")

    def select_skill(self, user_message: str) -> Optional[Skill]:
        """
        根据用户消息选择最合适的技能包。

        用 LLM 做意图匹配——分析用户问题，选择最相关的技能包。
        """
        if not self.skills:
            return None

        if len(self.skills) == 1:
            return list(self.skills.values())[0]

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
        return list(self.skills.values())[0]

    def get_skill_tools(self, skill: Skill) -> list:
        """
        获取技能包可用的工具列表。

        从所有 MCP Server 的工具中，筛选出该技能包需要的工具。
        如果技能包没有指定 required_tools，则返回所有可用工具。
        """
        available = self.mcp_client.discover_tools()

        if not skill.required_tools:
            return available

        skill_tool_names = set(skill.required_tools)
        return [t for t in available if t["name"] in skill_tool_names or not skill_tool_names]

    def validate_skill(self, skill: Skill) -> dict:
        """
        验证技能包的依赖是否满足。

        Returns:
            {"valid": bool, "missing_tools": [...], "missing_servers": [...]}
        """
        available_tools = self.mcp_client.discover_tools()
        available_tool_names = {t["name"] for t in available_tools}
        available_servers = set(self.mcp_client.servers.keys())

        missing_tools = [t for t in skill.required_tools if t not in available_tool_names]
        missing_servers = [s for s in skill.required_servers if s not in available_servers]

        return {
            "valid": not missing_tools and not missing_servers,
            "missing_tools": missing_tools,
            "missing_servers": missing_servers,
        }

    def list_skills(self) -> list:
        """列出所有已注册的技能包信息。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "required_tools": s.required_tools,
                "required_servers": s.required_servers,
            }
            for s in self.skills.values()
        ]


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第十章 示例 4：SkillManager 技能包管理器")
    print("=" * 60)

    # 创建 MCP Servers
    MOCK_ORDERS = {"ORD-001": {"product": "StarPods Pro", "status": "shipped"}}
    MOCK_LOGISTICS = {"ORD-001": {"courier": "顺丰", "location": "北京"}}
    MOCK_INVENTORY = {"StarPods Pro": {"stock": 156, "status": "有货"}}

    order_server = SimpleMCPServer("order_service", "订单管理服务")
    order_server.register_tool("query_order", "查询订单详情", lambda order_id: MOCK_ORDERS.get(order_id, {"error": "未找到"}))
    order_server.register_tool("query_logistics", "查询物流信息", lambda order_id: MOCK_LOGISTICS.get(order_id, {"error": "暂无"}))

    product_server = SimpleMCPServer("product_service", "产品信息服务")
    product_server.register_tool("query_inventory", "查询库存", lambda product_name: MOCK_INVENTORY.get(product_name, {"error": "未找到"}))
    product_server.register_tool("check_refund_policy", "查询退款政策", lambda product_name, order_days=0: {"policy": "7天退货" if order_days <= 7 else "质保期内"})

    # 创建技能包
    refund_skill = Skill(
        name="refund", description="处理退款、退货、换货请求",
        system_prompt="你是退款专员", required_servers=["order_service", "product_service"],
        required_tools=["query_order", "check_refund_policy"],
    )
    logistics_skill = Skill(
        name="logistics", description="查询物流和订单配送状态",
        system_prompt="你是物流专员", required_servers=["order_service"],
        required_tools=["query_order", "query_logistics"],
    )
    tech_skill = Skill(
        name="tech_support", description="处理产品故障和使用咨询",
        system_prompt="你是技术支持", required_servers=["order_service", "product_service"],
        required_tools=["query_order", "query_inventory"],
    )

    # 初始化 SkillManager
    print("\n--- 初始化 SkillManager ---")
    manager = SkillManager()
    manager.register_server(order_server)
    manager.register_server(product_server)
    manager.register_skill(refund_skill)
    manager.register_skill(logistics_skill)
    manager.register_skill(tech_skill)

    # 验证技能包
    print("\n--- 验证技能包依赖 ---")
    for skill_name, skill in manager.skills.items():
        validation = manager.validate_skill(skill)
        status = "通过" if validation["valid"] else f"缺少: {validation['missing_tools'] + validation['missing_servers']}"
        print(f"  {skill_name}: {status}")

    # 获取技能包工具
    print("\n--- 获取技能包工具 ---")
    for skill_name, skill in manager.skills.items():
        tools = manager.get_skill_tools(skill)
        tool_names = [t["name"] for t in tools]
        print(f"  {skill_name}: {tool_names}")

    # 技能包选择（简化演示，不调用 LLM）
    print("\n--- 技能包选择 ---")
    test_messages = [
        "帮我查一下快递到哪了",
        "我要退货",
        "耳机有杂音",
    ]
    for msg in test_messages:
        # 简化的关键词匹配选择（真实版用 LLM）
        if "退货" in msg or "退" in msg:
            selected = "refund"
        elif "快递" in msg or "物流" in msg or "到哪" in msg:
            selected = "logistics"
        else:
            selected = "tech_support"

        skill = manager.skills[selected]
        print(f"  消息: \"{msg}\" → 技能包: {skill.name} ({skill.description})")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. SkillManager 统一管理技能包和 MCP Server")
    print("  2. 技能包声明需要的工具和 Server，便于依赖检查")
    print("  3. select_skill() 根据用户消息自动匹配最合适的技能包")
    print("  4. get_skill_tools() 为技能包筛选可用工具")
    print("  5. validate_skill() 确保技能包的依赖都已就绪")
    print("=" * 60)
