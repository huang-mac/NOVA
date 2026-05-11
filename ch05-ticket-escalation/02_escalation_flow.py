"""
===============================================
第五章 示例 2：转人工流程
===============================================
目标：实现完整的转人工流程，包括排队、会话交接、回退机制。

知识点：
    1. 转人工的触发条件
    2. 会话交接协议（传递上下文摘要）
    3. 人工坐席队列管理
    4. 会话回退（人工处理完转回机器人）

运行方式：
    cd ch05-ticket-escalation
    python 02_escalation_flow.py
"""

import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 人工坐席模型
# ============================================================
class AgentStatus(str, Enum):
    AVAILABLE = "available"    # 空闲
    BUSY = "busy"              # 忙碌
    OFFLINE = "offline"        # 离线


class HumanAgent(BaseModel):
    """人工坐席。"""
    agent_id: str
    name: str
    status: AgentStatus = AgentStatus.AVAILABLE
    current_session: Optional[str] = None  # 当前处理的 session_id
    max_concurrent: int = 3                # 最大并发会话数
    active_sessions: list[str] = Field(default_factory=list)
    handled_count: int = 0                 # 已处理工单数


# ============================================================
# 转人工请求
# ============================================================
class EscalationRequest(BaseModel):
    """转人工请求。"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str
    user_id: str
    reason: str                        # 转人工原因
    context_summary: str               # 对话上下文摘要
    emotion_type: str = "neutral"
    priority: str = "normal"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    assigned_agent: Optional[str] = None
    status: str = "waiting"  # waiting / assigned / completed


# ============================================================
# 转人工管理器
# ============================================================
class EscalationManager:
    """
    转人工流程管理器。

    完整流程：
    1. 机器人判断需要转人工
    2. 创建转人工请求（附带上下文摘要）
    3. 排队等待可用坐席
    4. 分配坐席并交接会话
    5. 人工处理完成后回退到机器人（可选）
    """

    def __init__(self):
        self._agents: dict[str, HumanAgent] = {}
        self._queue: list[EscalationRequest] = []
        self._history: list[dict] = []

    def register_agent(self, agent_id: str, name: str) -> HumanAgent:
        """注册人工坐席。"""
        agent = HumanAgent(agent_id=agent_id, name=name)
        self._agents[agent_id] = agent
        return agent

    def request_escalation(
        self,
        session_id: str,
        user_id: str,
        reason: str,
        context_summary: str,
        emotion_type: str = "neutral",
        priority: str = "normal",
    ) -> EscalationRequest:
        """
        发起转人工请求。

        Args:
            session_id: 当前会话ID
            user_id: 用户ID
            reason: 转人工原因
            context_summary: 对话上下文摘要
            emotion_type: 用户情绪
            priority: 优先级

        Returns:
            EscalationRequest: 转人工请求
        """
        request = EscalationRequest(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
            context_summary=context_summary,
            emotion_type=emotion_type,
            priority=priority,
        )

        # 紧急请求插队到队首
        if priority == "urgent":
            self._queue.insert(0, request)
        else:
            self._queue.append(request)

        print(f"  📋 转人工请求已创建: {request.request_id}")
        print(f"     原因: {reason}")
        print(f"     当前排队: {len(self._queue)} 人")

        return request

    def assign_agent(self) -> Optional[tuple[EscalationRequest, HumanAgent]]:
        """
        分配空闲坐席给排队中的请求。

        Returns:
            tuple[EscalationRequest, HumanAgent] 或 None（无可用坐席或队列为空）
        """
        # 优先找空闲坐席，找不到再找未满的坐席
        available_agent = None
        for agent in self._agents.values():
            if agent.status == AgentStatus.AVAILABLE:
                available_agent = agent
                break

        if available_agent is None:
            for agent in self._agents.values():
                if len(agent.active_sessions) < agent.max_concurrent:
                    available_agent = agent
                    break

        if available_agent is None or not self._queue:
            return None

        # 取出队首请求
        request = self._queue.pop(0)

        # 分配
        request.assigned_agent = available_agent.agent_id
        request.status = "assigned"
        available_agent.active_sessions.append(request.session_id)
        available_agent.status = AgentStatus.BUSY

        print(f"  ✅ 坐席分配: {available_agent.name} ← 请求 {request.request_id}")

        # 记录历史
        self._history.append({
            "request_id": request.request_id,
            "agent_id": available_agent.agent_id,
            "assigned_at": datetime.now().isoformat(),
        })

        return request, available_agent

    def complete_session(self, session_id: str, agent_id: str, resolution: str):
        """标记会话处理完成。"""
        agent = self._agents.get(agent_id)
        if agent and session_id in agent.active_sessions:
            agent.active_sessions.remove(session_id)
            agent.handled_count += 1
            # 释放后有空位则恢复为可用
            if len(agent.active_sessions) < agent.max_concurrent:
                agent.status = AgentStatus.AVAILABLE

        print(f"  🏁 会话 {session_id} 处理完成")
        print(f"     处理结果: {resolution}")

    def get_handoff_context(self, request_id: str) -> Optional[dict]:
        """获取交接给人工坐席的上下文信息。"""
        for req in self._queue:
            if req.request_id == request_id:
                return {
                    "request_id": req.request_id,
                    "user_id": req.user_id,
                    "reason": req.reason,
                    "context_summary": req.context_summary,
                    "emotion_type": req.emotion_type,
                    "priority": req.priority,
                    "created_at": req.created_at,
                }
        return None

    def get_queue_status(self) -> dict:
        """获取当前队列状态。"""
        return {
            "queue_length": len(self._queue),
            "available_agents": len([
                a for a in self._agents.values()
                if a.status == AgentStatus.AVAILABLE
            ]),
            "total_agents": len(self._agents),
        }


def main():
    print("=" * 60)
    print("  第五章 示例 2：转人工流程")
    print("=" * 60)

    manager = EscalationManager()

    # --------------------------------------------------
    # 1. 注册人工坐席
    # --------------------------------------------------
    print("\n--- 注册坐席 ---\n")
    manager.register_agent("agent-001", "客服-小王")
    manager.register_agent("agent-002", "客服-小李")
    manager.register_agent("agent-003", "客服-小张")

    for agent in manager._agents.values():
        print(f"  ✓ {agent.name} ({agent.agent_id}) - {agent.status.value}")

    # --------------------------------------------------
    # 2. 模拟多个转人工请求
    # --------------------------------------------------
    print("\n--- 模拟转人工请求 ---\n")

    scenarios = [
        {
            "session_id": "session-001",
            "user_id": "user-001",
            "reason": "耳机质量问题，用户强烈不满",
            "context_summary": "用户张三反映 StarPods Pro 右耳有杂音，已尝试重置但仍存在问题。用户情绪愤怒，强度 9/10。",
            "emotion_type": "angry",
            "priority": "urgent",
        },
        {
            "session_id": "session-002",
            "user_id": "user-002",
            "reason": "订单物流问题",
            "context_summary": "用户李四的订单已下单5天未发货，多次催促。情绪烦躁，强度 6/10。",
            "emotion_type": "frustrated",
            "priority": "normal",
        },
        {
            "session_id": "session-003",
            "user_id": "user-003",
            "reason": "技术问题需要人工排查",
            "context_summary": "用户王五的 StarWatch X 屏幕偶发黑屏，已提供基础排查步骤但未解决。",
            "emotion_type": "confused",
            "priority": "normal",
        },
    ]

    requests = []
    for scenario in scenarios:
        req = manager.request_escalation(**scenario)
        requests.append(req)
        print()

    # --------------------------------------------------
    # 3. 坐席分配
    # --------------------------------------------------
    print("\n--- 坐席自动分配 ---\n")

    while True:
        status = manager.get_queue_status()
        if status["queue_length"] == 0:
            break

        print(f"  当前排队: {status['queue_length']} | 空闲坐席: {status['available_agents']}")
        result = manager.assign_agent()
        if result:
            req, agent = result

            # 展示交接信息（直接使用返回的 req 对象）
            print(f"\n  📋 交接信息 → {agent.name}:")
            print(f"     用户ID: {req.user_id}")
            print(f"     原因: {req.reason}")
            print(f"     上下文: {req.context_summary[:80]}...")
            print(f"     情绪: {req.emotion_type} | 优先级: {req.priority}")
            print()
        else:
            print("  ⏳ 无可用坐席，等待中...")
            break

    # --------------------------------------------------
    # 4. 模拟处理完成
    # --------------------------------------------------
    print("\n--- 模拟处理完成 ---\n")

    for req in requests:
        if req.assigned_agent:
            manager.complete_session(
                session_id=req.session_id,
                agent_id=req.assigned_agent,
                resolution="已为用户安排退货/已联系物流/已安排检修",
            )

    # 最终统计
    print("\n--- 坐席工作量统计 ---\n")
    for agent in manager._agents.values():
        print(f"  {agent.name}: 处理 {agent.handled_count} 个会话")

    print("\n✓ 第五章示例 2 运行完成！")
    print("  下一课：03_full_service.py - 完整售后流程集成")


if __name__ == "__main__":
    main()
