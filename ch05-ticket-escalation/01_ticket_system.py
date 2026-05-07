"""
===============================================
第五章 示例 1：工单数据模型与管理系统
===============================================
目标：设计并实现完整的工单管理系统，包括工单创建、状态流转、查询和统计。

知识点：
    1. Pydantic 数据模型设计（工单、工单记录、统计）
    2. 工单状态机设计（待处理 → 处理中 → 已解决 → 已关闭）
    3. JSON 文件持久化存储
    4. 工单编号自动生成

运行方式：
    cd ch05-ticket-escalation
    python 01_ticket_system.py
"""

import json
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 工单状态枚举（状态机）
# ============================================================
class TicketStatus(str, Enum):
    """
    工单状态流转：
    PENDING → PROCESSING → RESOLVED → CLOSED
      ↑           │
      └───────────┘  （退回重新处理）
    """
    PENDING = "pending"            # 待处理（刚创建）
    PROCESSING = "processing"      # 处理中（人工已接手）
    RESOLVED = "resolved"          # 已解决
    CLOSED = "closed"              # 已关闭
    REOPENED = "reopened"          # 重新打开


class TicketPriority(str, Enum):
    """工单优先级。"""
    LOW = "low"          # 低
    NORMAL = "normal"    # 普通
    HIGH = "high"        # 高
    URGENT = "urgent"    # 紧急


class TicketCategory(str, Enum):
    """工单分类。"""
    REFUND = "refund"            # 退款
    RETURN = "return"            # 退货
    EXCHANGE = "exchange"        # 换货
    REPAIR = "repair"            # 维修
    DELIVERY = "delivery"        # 物流问题
    COMPLAINT = "complaint"      # 投诉
    INQUIRY = "inquiry"          # 咨询
    OTHER = "other"              # 其他


# ============================================================
# 工单数据模型
# ============================================================
class TicketMessage(BaseModel):
    """工单中的消息记录。"""
    role: str = Field(description="发送者角色: user / agent / system")
    content: str = Field(description="消息内容")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Ticket(BaseModel):
    """
    工单数据模型。

    一个工单代表一个需要处理的用户问题/请求。
    """
    ticket_id: str = Field(description="工单编号，格式: TK-YYYYMMDD-XXXXX")
    category: TicketCategory = Field(description="工单分类")
    title: str = Field(description="工单标题（一句话概括问题）")
    description: str = Field(description="问题描述")
    priority: TicketPriority = Field(default=TicketPriority.NORMAL, description="优先级")
    status: TicketStatus = Field(default=TicketStatus.PENDING, description="当前状态")

    # 用户信息
    user_id: str = Field(description="用户标识")
    user_name: str = Field(default="", description="用户姓名")

    # 关联信息
    order_id: Optional[str] = Field(default=None, description="关联订单号")
    product: Optional[str] = Field(default=None, description="关联产品")
    emotion_type: Optional[str] = Field(default=None, description="用户情绪")
    chat_history: list[TicketMessage] = Field(
        default_factory=list, description="对话记录"
    )

    # 处理信息
    assigned_to: Optional[str] = Field(default=None, description="处理人")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="创建时间"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="最后更新时间"
    )
    resolved_at: Optional[str] = Field(default=None, description="解决时间")
    resolution: Optional[str] = Field(default=None, description="解决方案")


# ============================================================
# 工单管理系统
# ============================================================
class TicketManager:
    """
    工单管理系统。

    功能：
    - 创建工单
    - 更新工单状态
    - 查询工单
    - 工单统计
    - 持久化存储（JSON 文件）
    """

    def __init__(self, data_dir: str = "./data/tickets"):
        self.data_dir = data_dir
        self._tickets: dict[str, Ticket] = {}
        self._counter = 0

        # 确保存储目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 加载已有数据
        self._load_data()

    def _generate_ticket_id(self) -> str:
        """生成工单编号: TK-YYYYMMDD-XXXXX。"""
        date_str = datetime.now().strftime("%Y%m%d")
        self._counter += 1
        return f"TK-{date_str}-{self._counter:05d}"

    def create_ticket(
        self,
        category: TicketCategory,
        title: str,
        description: str,
        user_id: str,
        priority: TicketPriority = TicketPriority.NORMAL,
        user_name: str = "",
        order_id: Optional[str] = None,
        product: Optional[str] = None,
        emotion_type: Optional[str] = None,
        chat_history: Optional[list[TicketMessage]] = None,
    ) -> Ticket:
        """
        创建新工单。

        Args:
            category: 工单分类
            title: 工单标题
            description: 问题描述
            user_id: 用户ID
            priority: 优先级
            user_name: 用户姓名
            order_id: 订单号
            product: 产品名称
            emotion_type: 用户情绪
            chat_history: 对话记录

        Returns:
            Ticket: 创建的工单对象
        """
        ticket = Ticket(
            ticket_id=self._generate_ticket_id(),
            category=category,
            title=title,
            description=description,
            priority=priority,
            user_id=user_id,
            user_name=user_name,
            order_id=order_id,
            product=product,
            emotion_type=emotion_type,
            chat_history=chat_history or [],
        )

        self._tickets[ticket.ticket_id] = ticket
        self._save_data()
        return ticket

    def update_status(
        self,
        ticket_id: str,
        new_status: TicketStatus,
        assigned_to: Optional[str] = None,
        resolution: Optional[str] = None,
    ) -> Optional[Ticket]:
        """更新工单状态。"""
        if ticket_id not in self._tickets:
            return None

        ticket = self._tickets[ticket_id]

        # 验证状态流转合法性
        valid_transitions = {
            TicketStatus.PENDING: {TicketStatus.PROCESSING},
            TicketStatus.PROCESSING: {TicketStatus.RESOLVED, TicketStatus.PENDING},
            TicketStatus.RESOLVED: {TicketStatus.CLOSED, TicketStatus.REOPENED},
            TicketStatus.CLOSED: {TicketStatus.REOPENED},
            TicketStatus.REOPENED: {TicketStatus.PROCESSING},
        }

        if new_status not in valid_transitions.get(ticket.status, set()):
            print(f"  ⚠️ 非法状态流转: {ticket.status.value} → {new_status.value}")
            return None

        ticket.status = new_status
        ticket.updated_at = datetime.now().isoformat()

        if assigned_to:
            ticket.assigned_to = assigned_to
        if resolution:
            ticket.resolution = resolution
        if new_status == TicketStatus.RESOLVED:
            ticket.resolved_at = datetime.now().isoformat()

        # 记录状态变更消息
        ticket.chat_history.append(TicketMessage(
            role="system",
            content=f"工单状态变更为: {new_status.value}"
        ))

        self._save_data()
        return ticket

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """查询工单。"""
        return self._tickets.get(ticket_id)

    def list_tickets(
        self,
        user_id: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        category: Optional[TicketCategory] = None,
    ) -> list[Ticket]:
        """列出工单（支持过滤）。"""
        tickets = list(self._tickets.values())

        if user_id:
            tickets = [t for t in tickets if t.user_id == user_id]
        if status:
            tickets = [t for t in tickets if t.status == status]
        if category:
            tickets = [t for t in tickets if t.category == category]

        return sorted(tickets, key=lambda t: t.created_at, reverse=True)

    def get_statistics(self) -> dict:
        """获取工单统计信息。"""
        tickets = list(self._tickets.values())
        return {
            "total": len(tickets),
            "by_status": {
                status.value: len([t for t in tickets if t.status == status])
                for status in TicketStatus
            },
            "by_category": {
                cat.value: len([t for t in tickets if t.category == cat])
                for cat in TicketCategory
            },
            "by_priority": {
                pri.value: len([t for t in tickets if t.priority == pri])
                for pri in TicketPriority
            },
        }

    def _save_data(self):
        """持久化保存到 JSON 文件。"""
        file_path = os.path.join(self.data_dir, "tickets.json")
        data = {
            "counter": self._counter,
            "tickets": {tid: t.model_dump() for tid, t in self._tickets.items()},
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_data(self):
        """从 JSON 文件加载数据。"""
        file_path = os.path.join(self.data_dir, "tickets.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._counter = data.get("counter", 0)
            self._tickets = {
                tid: Ticket(**tdata)
                for tid, tdata in data.get("tickets", {}).items()
            }


def main():
    print("=" * 60)
    print("  第五章 示例 1：工单数据模型与管理系统")
    print("=" * 60)

    # 使用临时目录避免影响实际数据
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TicketManager(data_dir=tmpdir)

        # --------------------------------------------------
        # 1. 创建工单
        # --------------------------------------------------
        print("\n--- 创建工单 ---\n")

        t1 = manager.create_ticket(
            category=TicketCategory.RETURN,
            title="StarPods Pro 耳机退货申请",
            description="耳机右耳有杂音，购买两周内，申请退货",
            user_id="user-001",
            user_name="张三",
            order_id="ORD-20240115-001",
            product="StarPods Pro",
            priority=TicketPriority.NORMAL,
            emotion_type="frustrated",
        )
        print(f"✓ 工单已创建: {t1.ticket_id}")
        print(f"  分类: {t1.category.value} | 优先级: {t1.priority.value}")
        print(f"  标题: {t1.title}")

        t2 = manager.create_ticket(
            category=TicketCategory.COMPLAINT,
            title="多次投诉未解决",
            description="已三次联系客服但问题仍未解决，非常不满",
            user_id="user-002",
            user_name="李四",
            priority=TicketPriority.URGENT,
            emotion_type="angry",
        )
        print(f"\n✓ 工单已创建: {t2.ticket_id}")
        print(f"  分类: {t2.category.value} | 优先级: {t2.priority.value}")

        t3 = manager.create_ticket(
            category=TicketCategory.INQUIRY,
            title="StarWatch X 使用咨询",
            description="询问手表血氧监测功能的使用方法",
            user_id="user-003",
            product="StarWatch X",
        )
        print(f"\n✓ 工单已创建: {t3.ticket_id}")

        # --------------------------------------------------
        # 2. 状态流转
        # --------------------------------------------------
        print("\n\n--- 工单状态流转 ---\n")

        # t1: 待处理 → 处理中
        manager.update_status(t1.ticket_id, TicketStatus.PROCESSING, assigned_to="客服-小王")
        print(f"[{t1.ticket_id}] {t1.status.value} → 已分配给 客服-小王")

        # t1: 处理中 → 已解决
        manager.update_status(
            t1.ticket_id, TicketStatus.RESOLVED,
            resolution="已为客户安排退货，顺丰取件中"
        )
        print(f"[{t1.ticket_id}] {t1.status.value} → 已解决")

        # 测试非法状态流转
        print(f"\n[非法流转测试] 待处理 → 已关闭:")
        t_new = manager.create_ticket(
            category=TicketCategory.OTHER,
            title="测试工单",
            description="测试非法状态流转",
            user_id="test",
        )
        result = manager.update_status(t_new.ticket_id, TicketStatus.CLOSED)
        print(f"  结果: {'被拦截 ✓' if result is None else '未拦截 ✗'}")

        # --------------------------------------------------
        # 3. 查询与统计
        # --------------------------------------------------
        print("\n\n--- 工单查询与统计 ---\n")

        stats = manager.get_statistics()
        print(f"工单总数: {stats['total']}")
        print(f"\n按状态分布:")
        for status, count in stats["by_status"].items():
            if count > 0:
                print(f"  {status}: {count}")
        print(f"\n按分类分布:")
        for cat, count in stats["by_category"].items():
            if count > 0:
                print(f"  {cat}: {count}")

        # 查询特定用户工单
        user_tickets = manager.list_tickets(user_id="user-001")
        print(f"\n张三的工单: {len(user_tickets)} 个")
        for t in user_tickets:
            print(f"  {t.ticket_id} | {t.status.value:12s} | {t.title}")

        print("\n✓ 第五章示例 1 运行完成！")
        print("  下一课：02_escalation_flow.py - 转人工流程")


if __name__ == "__main__":
    main()
