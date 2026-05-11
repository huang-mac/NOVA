"""
===============================================
第十章 示例 2：MCP Client 调用工具
===============================================
目标：实现一个 MCP Client，能连接多个 MCP Server，
     发现所有可用工具，并统一调用。

MCP Client 是 AI 应用内的"工具管理器"——连接 Server、发现工具、调用工具。
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

# 从上一节导入 MCP Server
from typing import Optional


# ============================================================
# 简化 MCP Server（内联，不依赖 01 文件）
# ============================================================
class SimpleMCPServer:
    """简化的 MCP Server。"""

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


# ============================================================
# MCP Client
# ============================================================
class SimpleMCPClient:
    """
    简化的 MCP Client。

    职责：
    1. 连接多个 MCP Server
    2. 发现所有 Server 上的可用工具
    3. 统一工具调用接口（不用关心工具在哪个 Server 上）
    """

    def __init__(self):
        self.servers = {}  # name → SimpleMCPServer

    def connect(self, server: SimpleMCPServer):
        """
        连接一个 MCP Server。

        Args:
            server: MCP Server 实例
        """
        self.servers[server.name] = server
        print(f"  [MCP Client] 已连接 Server: {server.name} ({len(server.tools)} 个工具)")

    def disconnect(self, server_name: str):
        """断开与指定 Server 的连接。"""
        if server_name in self.servers:
            del self.servers[server_name]
            print(f"  [MCP Client] 已断开 Server: {server_name}")

    def discover_tools(self) -> list:
        """
        从所有已连接的 Server 发现可用工具。

        Returns:
            工具列表，每个元素包含 name、description、server
        """
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.list_tools())
        return all_tools

    def get_tool_info(self, tool_name: str) -> Optional[dict]:
        """获取工具的详细信息。"""
        for server in self.servers.values():
            for tool in server.list_tools():
                if tool["name"] == tool_name:
                    return tool
        return None

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """
        调用指定工具（自动查找工具所在的 Server）。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        for server in self.servers.values():
            if tool_name in server.tools:
                print(f"  [MCP Client] 调用 {server.name}.{tool_name}")
                return server.call_tool(tool_name, arguments)
        return json.dumps({"error": f"未找到工具 {tool_name}"}, ensure_ascii=False)

    def status(self) -> dict:
        """返回 Client 状态摘要。"""
        return {
            "connected_servers": len(self.servers),
            "server_names": list(self.servers.keys()),
            "total_tools": len(self.discover_tools()),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第十章 示例 2：MCP Client 调用工具")
    print("=" * 60)

    # 创建两个 MCP Server
    print("\n--- 创建 MCP Servers ---")
    order_server = SimpleMCPServer("order_service", "订单管理服务")
    order_server.register_tool("query_order", "查询订单详情",
        lambda order_id: {"product": "StarPods Pro", "status": "shipped"} if order_id == "ORD-001" else {"error": "未找到"})
    order_server.register_tool("query_logistics", "查询物流信息",
        lambda order_id: {"courier": "顺丰", "location": "北京"} if order_id == "ORD-001" else {"error": "暂无"})

    product_server = SimpleMCPServer("product_service", "产品信息服务")
    product_server.register_tool("query_inventory", "查询商品库存",
        lambda product_name: {"stock": 156, "status": "有货"} if "Pro" in product_name else {"stock": 0, "status": "缺货"})
    product_server.register_tool("check_refund_policy", "查询退款政策",
        lambda product_name, order_days=0: {"policy": "7天退货" if order_days <= 7 else "质保期内"})

    # 创建 Client 并连接
    print("\n--- Client 连接 Servers ---")
    client = SimpleMCPClient()
    client.connect(order_server)
    client.connect(product_server)

    # 发现工具
    print("\n--- 发现工具 ---")
    tools = client.discover_tools()
    print(f"  共发现 {len(tools)} 个工具：")
    for t in tools:
        print(f"    - {t['name']} ({t['server']}): {t['description']}")

    # 调用工具
    print("\n--- 调用工具 ---")
    print(f"  [1] query_order:")
    result = client.call_tool("query_order", {"order_id": "ORD-001"})
    print(f"      结果: {result}")

    print(f"  [2] query_inventory:")
    result = client.call_tool("query_inventory", {"product_name": "StarPods Pro"})
    print(f"      结果: {result}")

    print(f"  [3] query_inventory (缺货):")
    result = client.call_tool("query_inventory", {"product_name": "StarPods Lite"})
    print(f"      结果: {result}")

    # 错误处理
    print("\n--- 错误处理 ---")
    result = client.call_tool("nonexistent_tool", {})
    print(f"  不存在的工具: {result}")

    # 状态
    print(f"\n--- Client 状态 ---")
    print(f"  {json.dumps(client.status(), ensure_ascii=False)}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. MCP Client 是 Agent 和 MCP Server 之间的桥梁")
    print("  2. Client 连接多个 Server，对 Agent 来说工具是统一的")
    print("  3. discover_tools() 让 Agent 知道有哪些工具可用")
    print("  4. call_tool() 自动路由到正确的 Server，Agent 不需要关心工具在哪")
    print("  5. 新增一个 Server 只需 connect() 一次，Agent 代码不用改")
    print("=" * 60)
