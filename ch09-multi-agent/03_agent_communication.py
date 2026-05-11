"""
===============================================
第九章 示例 3：Agent 间通信与任务委派
===============================================
目标：展示 Agent 之间如何通信、传递上下文和委派任务。

在 Multi-Agent 系统中，Agent 之间的通信方式决定了系统的灵活性和可扩展性。
本章采用"直接消息传递 + 共享状态"的混合方式。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm


# ============================================================
# 方式一：消息传递（Message Passing）
# ============================================================
class AgentMessage:
    """
    Agent 间传递的消息对象。

    类似于公司内部的工单——前台接待写好用户需求和已知信息，
    附在工单上交给对应的专员处理。
    """

    def __init__(self, sender: str, receiver: str,
                 user_message: str, context: dict = None,
                 priority: str = "normal"):
        self.sender = sender              # 发送方 Agent 名
        self.receiver = receiver          # 接收方 Agent 名
        self.user_message = user_message  # 用户原始消息
        self.context = context or {}      # 附加上下文（如提取的订单号等）
        self.priority = priority          # 优先级：normal / high / urgent

    def to_prompt_context(self) -> str:
        """将消息转换为可注入 Prompt 的上下文字符串。"""
        parts = [f"来自 {self.sender} 的转派消息："]
        if self.context:
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")
        parts.append(f"用户原始消息：{self.user_message}")
        return "\n".join(parts)

    def __repr__(self):
        return f"AgentMessage({self.sender} → {self.receiver}, priority={self.priority})"


# ============================================================
# 方式二：共享状态（Shared State）
# ============================================================
class ConversationState(TypedDict):
    """
    Agent 之间共享的对话状态。

    类似于客服系统中的"工单记录"——所有 Agent 都能看到之前的处理过程。
    """
    session_id: str           # 会话 ID
    user_id: str              # 用户 ID
    messages: list            # 对话历史
    current_agent: str        # 当前处理的 Agent
    routing_history: list     # 路由历史 [{"agent": ..., "timestamp": ...}]
    collected_info: dict      # 各 Agent 收集到的信息
    status: str               # 工单状态：open / processing / resolved / escalated


class SharedStateStore:
    """共享状态存储（简化版，真实场景用 Redis 或数据库）。"""

    def __init__(self):
        self._states = {}

    def create(self, session_id: str, user_id: str) -> ConversationState:
        """创建新的会话状态。"""
        state = ConversationState(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            current_agent="reception",
            routing_history=[{"agent": "reception", "action": "新建会话"}],
            collected_info={},
            status="open",
        )
        self._states[session_id] = state
        return state

    def get(self, session_id: str) -> Optional[ConversationState]:
        return self._states.get(session_id)

    def update(self, session_id: str, **updates) -> ConversationState:
        """更新会话状态。"""
        state = self._states.get(session_id)
        if state:
            state.update(updates)
        return state


# ============================================================
# 任务委派演示
# ============================================================
def demo_message_passing():
    """演示消息传递方式。"""
    print("\n  --- 方式一：消息传递 ---")

    # 接待 Agent 创建消息，转派给售后
    msg = AgentMessage(
        sender="reception",
        receiver="after_sales",
        user_message="我要退订单 ORD-20240125-001，买了 10 天了",
        context={"order_id": "ORD-20240125-001", "product": "StarPods Pro", "intent": "refund"},
        priority="normal",
    )

    print(f"  消息：{msg}")
    print(f"  转化为 Prompt 上下文：\n{msg.to_prompt_context()}")


def demo_shared_state():
    """演示共享状态方式。"""
    print("\n  --- 方式二：共享状态 ---")

    store = SharedStateStore()

    # 创建会话
    state = store.create("session-001", "user-123")
    print(f"  初始状态：agent={state['current_agent']}, status={state['status']}")

    # 接待 Agent 路由后更新状态
    state = store.update(
        "session-001",
        current_agent="after_sales",
        routing_history=state["routing_history"] + [{"agent": "reception", "action": "路由到售后"}],
        status="processing",
        collected_info={"order_id": "ORD-20240125-001", "product": "StarPods Pro"},
    )
    print(f"  路由后：agent={state['current_agent']}, status={state['status']}")

    # 售后 Agent 处理后更新
    state = store.update(
        "session-001",
        routing_history=state["routing_history"] + [{"agent": "after_sales", "action": "已处理退款咨询"}],
        collected_info={**state["collected_info"], "refund_eligible": False, "warranty_active": True},
        status="resolved",
    )
    print(f"  处理后：agent={state['current_agent']}, status={state['status']}")
    print(f"  收集的信息：{json.dumps(state['collected_info'], ensure_ascii=False)}")
    print(f"  路由历史：{state['routing_history']}")


def demo_combined_approach():
    """演示混合方式：消息传递 + 共享状态。"""
    print("\n  --- 混合方式：消息传递 + 共享状态 ---")

    store = SharedStateStore()
    state = store.create("session-002", "user-456")

    # Step 1: 接收消息
    user_message = "我的耳机连不上手机了"
    print(f"\n  [1] 用户：{user_message}")

    # Step 2: 接待 Agent 分析 + 路由
    print(f"  [2] 接待 Agent 分析意图...")

    # 模拟路由决策（真实场景用 LLM）
    route_decision = {"agent": "tech_support", "reason": "用户反馈连接问题"}

    # Step 3: 创建消息 + 更新共享状态
    msg = AgentMessage(
        sender="reception",
        receiver=route_decision["agent"],
        user_message=user_message,
        context={"intent": "tech_issue"},
    )

    store.update(
        "session-002",
        current_agent=route_decision["agent"],
        routing_history=state["routing_history"] + [
            {"agent": "reception", "action": f"路由到 {route_decision['agent']}"}
        ],
        status="processing",
    )

    print(f"  [3] 消息：{msg}")
    print(f"  [3] 共享状态更新：agent={store.get('session-002')['current_agent']}")

    # Step 4: 专业 Agent 从消息中获取上下文
    context_prompt = msg.to_prompt_context()
    print(f"  [4] 技术 Agent 收到的上下文：\n{context_prompt}")


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("第九章 示例 3：Agent 间通信与任务委派")
    print("=" * 60)

    demo_message_passing()
    demo_shared_state()
    demo_combined_approach()

    print("\n" + "=" * 60)
    print("关键要点：")
    print("  1. 消息传递：Agent 之间通过消息对象通信，清晰但耦合度稍高")
    print("  2. 共享状态：所有 Agent 读写同一份状态，适合紧密协作")
    print("  3. 事件总线（本章未实现）：松耦合，Agent 发布/订阅事件")
    print("  4. 混合方式 = 消息传递做路由 + 共享状态做信息共享")
    print("  5. 路由历史记录很重要，用于问题追溯和质检")
    print("=" * 60)
