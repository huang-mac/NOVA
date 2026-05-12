"""
===============================================
第七章 示例 2：客服场景的 Tool 实现
===============================================
目标：定义一组客服常用的工具（查订单、查物流、查库存），模拟真实业务数据。

实际落地时，把这里的 mock 数据替换成真实的数据库查询或 API 调用即可。
"""

import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.tools import tool


# ============================================================
# 模拟数据 —— 实际项目中替换成真实数据库查询
# ============================================================
def load_mock_orders():
    """加载模拟订单数据。"""
    return {
        "ORD-20240125-001": {
            "order_id": "ORD-20240125-001",
            "user_id": "U10086",
            "product_name": "StarPods Pro 降噪耳机",
            "quantity": 1,
            "price": 899.00,
            "status": "shipped",        # pending / paid / shipped / delivered / cancelled
            "create_time": "2024-01-25 14:30:00",
            "ship_time": "2024-01-26 09:15:00",
        },
        "ORD-20240128-002": {
            "order_id": "ORD-20240128-002",
            "user_id": "U10086",
            "product_name": "StarWatch S1 智能手表",
            "quantity": 2,
            "price": 1299.00,
            "status": "pending",
            "create_time": "2024-01-28 20:00:00",
            "ship_time": None,
        },
        "ORD-20240120-003": {
            "order_id": "ORD-20240120-003",
            "user_id": "U10086",
            "product_name": "StarCharge 充电宝 20000mAh",
            "quantity": 1,
            "price": 199.00,
            "status": "delivered",
            "create_time": "2024-01-20 10:00:00",
            "ship_time": "2024-01-21 08:00:00",
            "deliver_time": "2024-01-23 15:30:00",
        },
    }


def load_mock_logistics():
    """加载模拟物流数据。"""
    return {
        "ORD-20240125-001": {
            "order_id": "ORD-20240125-001",
            "courier": "顺丰速运",
            "tracking_number": "SF1234567890",
            "current_location": "已到达北京分拨中心",
            "estimated_delivery": "2024-01-28",
            "history": [
                {"time": "01-26 09:15", "status": "已揽收", "location": "深圳仓库"},
                {"time": "01-26 18:00", "status": "运输中", "location": "深圳 → 北京"},
                {"time": "01-27 06:00", "status": "已到达", "location": "北京分拨中心"},
            ]
        },
    }


def load_mock_inventory():
    """加载模拟库存数据。"""
    return {
        "StarPods Pro": {
            "product_name": "StarPods Pro 降噪耳机",
            "price": 899.00,
            "stock": 156,
            "warehouse": "深圳主仓",
            "status": "有货",
        },
        "StarPods Lite": {
            "product_name": "StarPods Lite 蓝牙耳机",
            "price": 299.00,
            "stock": 0,
            "warehouse": "深圳主仓",
            "status": "缺货",
        },
        "StarWatch S1": {
            "product_name": "StarWatch S1 智能手表",
            "price": 1299.00,
            "stock": 89,
            "warehouse": "深圳主仓",
            "status": "有货",
        },
        "StarCharge": {
            "product_name": "StarCharge 充电宝 20000mAh",
            "price": 199.00,
            "stock": 3,
            "warehouse": "深圳主仓",
            "status": "库存紧张",
        },
    }


# ============================================================
# 定义客服工具
# ============================================================
@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单详情，包括商品信息、金额、状态、下单时间等。
    当用户询问某个订单的状态、内容或详情时使用此工具。

    Args:
        order_id: 订单号，格式为 ORD-xxxxxxxx-xxx，例如 ORD-20240125-001
    """
    orders = load_mock_orders()
    order = orders.get(order_id)
    if order:
        return json.dumps(order, ensure_ascii=False, indent=2)
    return json.dumps({
        "error": f"未找到订单 {order_id}",
        "suggestion": "请确认订单号是否正确。可以在 APP 的'我的订单'中查看。"
    }, ensure_ascii=False)


@tool
def query_logistics(order_id: str) -> str:
    """查询订单的物流信息，包括快递公司、运单号、当前位置、预计到达时间等。
    当用户询问包裹到哪了、什么时候送到、快递单号是多少时使用此工具。

    Args:
        order_id: 订单号
    """
    logistics = load_mock_logistics()
    info = logistics.get(order_id)
    if info:
        return json.dumps(info, ensure_ascii=False, indent=2)
    return json.dumps({
        "error": f"未找到订单 {order_id} 的物流信息",
        "suggestion": "可能还未发货，您可以先查询订单状态确认是否已发货。"
    }, ensure_ascii=False)


@tool
def query_inventory(product_name: str) -> str:
    """查询商品库存信息。
    当用户询问某个商品是否有货、库存多少、什么时候有货时使用此工具。

    Args:
        product_name: 商品名称，例如 StarPods Pro、StarWatch S1、StarPods Lite
    """
    inventory = load_mock_inventory()
    # 模糊匹配：如果用户输入的名字不完全匹配，尝试找最接近的
    for name, item in inventory.items():
        if product_name.lower() in name.lower() or name.lower() in product_name.lower():
            return json.dumps(item, ensure_ascii=False, indent=2)
    return json.dumps({
        "error": f"未找到商品 '{product_name}'",
        "suggestion": f"目前在售的商品有：{', '.join(inventory.keys())}"
    }, ensure_ascii=False)


@tool
def check_refund_policy(product_name: str, order_days: int = 0) -> str:
    """查询退款/退货政策。
    当用户询问能不能退货、退款规则是什么、退货期限是多少天时使用此工具。

    Args:
        product_name: 商品名称
        order_days: 下单已过天数，如果用户提到了就传，没提到默认 0
    """
    base_policy = {
        "standard_return": "7 天无理由退货",
        "quality_warranty": "1 年质保",
        "non_returnable": ["定制商品", "已拆封的贴身物品"],
        "process": "APP 申请 → 审核通过 → 寄回商品 → 收到后 3 个工作日退款"
    }

    # 如果用户提到了下单天数，顺便给个具体判断
    extra = ""
    if order_days > 0:
        if order_days <= 7:
            extra = f"\n您的商品下单 {order_days} 天，仍在 7 天无理由退货期内，可以正常申请退货。"
        else:
            extra = f"\n您的商品下单 {order_days} 天，已超出 7 天无理由退货期，但仍在 1 年质保期内，如遇质量问题可以申请售后维修。"

    return json.dumps({
        "product": product_name,
        "policy": base_policy,
    }, ensure_ascii=False, indent=2) + extra


# ============================================================
# 获取所有客服工具
# ============================================================
def get_customer_tools():
    """获取所有客服场景的工具列表。"""
    return [query_order, query_logistics, query_inventory, check_refund_policy]


# ============================================================
# 主函数：测试工具
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第七章 示例 2：客服场景 Tool 测试")
    print("=" * 60)

    tools = get_customer_tools()
    print(f"\n已定义 {len(tools)} 个客服工具：")
    for t in tools:
        print(f"  - {t.name}: {t.description[:50]}...")

    # 测试查订单
    print("\n--- 测试查订单 ---")
    result = query_order.invoke({"order_id": "ORD-20240125-001"})
    print(result)

    # 测试查物流
    print("\n--- 测试查物流 ---")
    result = query_logistics.invoke({"order_id": "ORD-20240125-001"})
    print(result)

    # 测试查库存
    print("\n--- 测试查库存 ---")
    result = query_inventory.invoke({"product_name": "StarPods Pro"})
    print(result)

    # 测试退款政策
    print("\n--- 测试退款政策 ---")
    result = check_refund_policy.invoke({"product_name": "StarPods Pro", "order_days": 10})
    print(result)

    # 测试查不到的情况
    print("\n--- 测试查不到的订单 ---")
    result = query_order.invoke({"order_id": "ORD-99999999-999"})
    print(result)

    print("\n" + "=" * 60)
    print("要点：工具内部逻辑可以随意替换，LLM 看到的是 name + docstring")
    print("实际项目中，把 mock 数据换成真实数据库查询就行")
    print("=" * 60)
