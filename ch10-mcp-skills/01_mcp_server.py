"""
===============================================
第十章 示例 1：实现一个简单的 MCP Server
===============================================
目标：理解 MCP Server 的核心概念，实现一个简化版的 MCP Server。

MCP（Model Context Protocol）是 AI 领域的"USB 接口"——
工具提供方实现一次 MCP Server，所有支持 MCP 的 AI 应用直接用。
"""

import json


# ============================================================
# 简化 MCP Server
# ============================================================
class SimpleMCPServer:
    """
    简化的 MCP Server 实现。

    真实的 MCP Server 需要：
    - 实现 JSON-RPC 2.0 协议
    - 支持 tools/list、tools/call 等 MCP 标准方法
    - 通过 stdio 或 HTTP/SSE 通信
    - 支持资源（resources）、提示（prompts）等扩展

    这里用简化版展示核心逻辑：工具注册 + 请求处理。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools = {}  # tool_name → {name, description, handler, parameters}

    def register_tool(self, name: str, description: str, handler: callable,
                      parameters: dict = None):
        """
        注册一个工具。

        Args:
            name: 工具名称（全局唯一）
            description: 工具功能描述
            handler: 工具的处理函数
            parameters: 参数 schema（JSON Schema 格式）
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters or {},
        }
        print(f"  [MCP Server:{self.name}] 注册工具: {name}")

    def list_tools(self) -> list:
        """
        列出所有已注册的工具（对应 MCP 协议的 tools/list 方法）。

        Returns:
            工具列表，每个元素包含 name、description、server
        """
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
                "server": self.name,
            }
            for t in self.tools.values()
        ]

    def call_tool(self, tool_name: str, arguments: dict = None) -> str:
        """
        调用指定工具（对应 MCP 协议的 tools/call 方法）。

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果（JSON 字符串）
        """
        arguments = arguments or {}
        tool = self.tools.get(tool_name)

        if not tool:
            return json.dumps({"error": f"工具 {tool_name} 不存在"}, ensure_ascii=False)

        try:
            result = tool["handler"](**arguments)
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except TypeError as e:
            return json.dumps({"error": f"参数错误: {e}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"执行错误: {e}"}, ensure_ascii=False)

    def handle_request(self, method: str, params: dict = None) -> dict:
        """
        处理 MCP 请求（模拟 JSON-RPC 协议）。

        Args:
            method: MCP 方法名（tools/list 或 tools/call）
            params: 请求参数

        Returns:
            MCP 标准响应
        """
        params = params or {}

        if method == "tools/list":
            return {"tools": self.list_tools()}
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = self.call_tool(tool_name, arguments)
            return {"content": [{"type": "text", "text": result}]}
        else:
            return {"error": f"未知方法: {method}"}

    def info(self) -> dict:
        """返回 Server 信息。"""
        return {
            "name": self.name,
            "description": self.description,
            "tool_count": len(self.tools),
            "tools": list(self.tools.keys()),
        }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第十章 示例 1：实现一个简单的 MCP Server")
    print("=" * 60)

    # 创建订单服务 MCP Server
    print("\n--- 创建订单服务 Server ---")
    order_server = SimpleMCPServer("order_service", "订单管理服务")

    # 注册工具
    MOCK_ORDERS = {
        "ORD-20240125-001": {"product": "StarPods Pro", "status": "shipped", "price": 899},
    }

    def query_order(order_id: str) -> dict:
        """查询订单详情。"""
        return MOCK_ORDERS.get(order_id, {"error": "未找到"})

    def query_logistics(order_id: str) -> dict:
        """查询物流信息。"""
        if order_id == "ORD-20240125-001":
            return {"courier": "顺丰", "location": "北京分拨中心", "eta": "明天"}
        return {"error": "暂无物流"}

    order_server.register_tool("query_order", "查询订单详情", query_order,
                               {"type": "object", "properties": {"order_id": {"type": "string"}}})
    order_server.register_tool("query_logistics", "查询物流信息", query_logistics,
                               {"type": "object", "properties": {"order_id": {"type": "string"}}})

    print(f"\n  Server 信息：{json.dumps(order_server.info(), ensure_ascii=False)}")

    # 测试 tools/list
    print("\n--- 测试 tools/list ---")
    tools = order_server.handle_request("tools/list")
    print(f"  可用工具：{json.dumps(tools, ensure_ascii=False, indent=2)}")

    # 测试 tools/call
    print("\n--- 测试 tools/call ---")
    result = order_server.handle_request("tools/call", {
        "name": "query_order",
        "arguments": {"order_id": "ORD-20240125-001"},
    })
    print(f"  query_order 结果：{json.dumps(result, ensure_ascii=False)}")

    result = order_server.handle_request("tools/call", {
        "name": "query_logistics",
        "arguments": {"order_id": "ORD-20240125-001"},
    })
    print(f"  query_logistics 结果：{json.dumps(result, ensure_ascii=False)}")

    # 测试错误处理
    print("\n--- 测试错误处理 ---")
    result = order_server.handle_request("tools/call", {
        "name": "not_exist_tool",
        "arguments": {},
    })
    print(f"  不存在的工具：{json.dumps(result, ensure_ascii=False)}")

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. MCP Server 本质是「工具注册 + 请求处理」")
    print("  2. 核心方法：tools/list（列出工具）、tools/call（调用工具）")
    print("  3. 每个 MCP Server 专注一个业务领域（如订单、物流、CRM）")
    print("  4. 真实的 MCP Server 使用 JSON-RPC 2.0 协议通信")
    print("  5. 工具的参数用 JSON Schema 描述，让 Client 知道该传什么参数")
    print("=" * 60)
