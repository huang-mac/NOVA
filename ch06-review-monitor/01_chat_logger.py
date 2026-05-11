"""
===============================================
第六章 示例 1：聊天记录存储模块
===============================================
目标：实现完整的聊天记录存储系统，支持按会话、用户、时间查询。

知识点：
    1. 对话记录的数据模型设计
    2. JSON 文件持久化
    3. 按多维度查询
    4. 数据导出

运行方式：
    cd ch06-review-monitor
    python 01_chat_logger.py
"""

import json
import os
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================
# 数据模型
# ============================================================
class MessageRole(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """单条聊天消息。"""
    role: MessageRole
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = Field(default_factory=dict)


class ConversationSession(BaseModel):
    """一次完整的对话会话。"""
    session_id: str
    user_id: str
    user_name: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    messages: list[ChatMessage] = Field(default_factory=list)
    status: str = "active"  # active / closed / escalated

    # 会话元数据
    total_messages: int = 0
    emotion_changes: list[dict] = Field(default_factory=list)
    routing_decisions: list[dict] = Field(default_factory=list)
    ticket_id: Optional[str] = None
    escalation_id: Optional[str] = None

    def add_message(self, role: MessageRole, content: str, **metadata):
        """添加消息到会话。"""
        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        self.total_messages += 1
        self.updated_at = datetime.now().isoformat()

    def to_summary(self) -> dict:
        """生成会话摘要。"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "created_at": self.created_at,
            "total_messages": self.total_messages,
            "status": self.status,
            "ticket_id": self.ticket_id,
            "escalation_id": self.escalation_id,
            "first_message": self.messages[0].content[:50] if self.messages else "",
            "last_message": self.messages[-1].content[:50] if self.messages else "",
        }


# ============================================================
# 聊天记录管理器
# ============================================================
class ChatLogger:
    """
    聊天记录管理器。

    功能：
    - 记录每条消息
    - 管理会话生命周期
    - 按维度查询历史记录
    - 数据导出
    """

    def __init__(self, data_dir: str = "./data/chat_logs"):
        self.data_dir = data_dir
        self._sessions: dict[str, ConversationSession] = {}
        os.makedirs(data_dir, exist_ok=True)
        self._load_data()

    def create_session(self, session_id: str, user_id: str, user_name: str = "") -> ConversationSession:
        """创建新会话。"""
        session = ConversationSession(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
        )
        self._sessions[session_id] = session
        session.add_message(
            MessageRole.SYSTEM,
            "会话开始",
            event="session_start",
        )
        self._save_data()
        return session

    def log_message(self, session_id: str, role: MessageRole, content: str, **metadata):
        """记录一条消息。"""
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"会话 {session_id} 不存在")

        session.add_message(role, content, **metadata)
        self._save_data()

    def close_session(self, session_id: str, status: str = "closed"):
        """关闭会话。"""
        session = self._sessions.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now().isoformat()
            session.add_message(MessageRole.SYSTEM, f"会话结束: {status}", event="session_end")
            self._save_data()

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取完整会话记录。"""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> list[ConversationSession]:
        """按条件查询会话列表。"""
        sessions = list(self._sessions.values())

        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        if date_from:
            sessions = [s for s in sessions if s.created_at >= date_from]
        if date_to:
            sessions = [s for s in sessions if s.created_at <= date_to]

        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def get_statistics(self) -> dict:
        """获取整体统计数据。"""
        sessions = list(self._sessions.values())
        total_messages = sum(s.total_messages for s in sessions)

        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "active_sessions": len([s for s in sessions if s.status == "active"]),
            "closed_sessions": len([s for s in sessions if s.status == "closed"]),
            "escalated_sessions": len([s for s in sessions if s.status == "escalated"]),
            "sessions_with_tickets": len([s for s in sessions if s.ticket_id]),
            "avg_messages_per_session": total_messages / len(sessions) if sessions else 0,
        }

    def export_session(self, session_id: str, format: str = "json") -> str:
        """导出单个会话记录。"""
        session = self._sessions.get(session_id)
        if not session:
            return ""

        if format == "text":
            lines = [
                f"会话ID: {session.session_id}",
                f"用户: {session.user_name or session.user_id}",
                f"时间: {session.created_at}",
                f"消息数: {session.total_messages}",
                f"状态: {session.status}",
                "-" * 50,
            ]
            for msg in session.messages:
                time_str = msg.timestamp[:19]
                role_str = {"user": "用户", "agent": "小星", "system": "系统"}.get(msg.role.value, msg.role.value)
                lines.append(f"[{time_str}] {role_str}: {msg.content}")
            return "\n".join(lines)
        else:
            return session.model_dump_json(indent=2)

    def _save_data(self):
        """保存数据到磁盘。"""
        file_path = os.path.join(self.data_dir, "sessions.json")
        data = {
            tid: s.model_dump()
            for tid, s in self._sessions.items()
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_data(self):
        """从磁盘加载数据。"""
        file_path = os.path.join(self.data_dir, "sessions.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._sessions = {
                tid: ConversationSession(**sdata)
                for tid, sdata in data.items()
            }


def main():
    print("=" * 60)
    print("  第六章 示例 1：聊天记录存储模块")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ChatLogger(data_dir=tmpdir)

        # --- 创建会话并记录消息 ---
        print("\n--- 创建会话并记录消息 ---\n")

        s1 = logger.create_session("session-001", "user-alice", "Alice")
        logger.log_message("session-001", MessageRole.USER, "你好，我想了解一下 StarPods Pro")
        logger.log_message("session-001", MessageRole.AGENT, "您好 Alice！StarPods Pro 是我们的旗舰降噪耳机...", source="rag")
        logger.log_message("session-001", MessageRole.USER, "降噪效果怎么样？")
        logger.log_message("session-001", MessageRole.AGENT, "StarPods Pro 搭载 StarNoise 3.0 芯片，支持最高 48dB 主动降噪。", source="rag")
        logger.log_message("session-001", MessageRole.USER, "好的，谢谢！")
        logger.log_message("session-001", MessageRole.AGENT, "不客气，还有其他问题随时找我~")
        logger.close_session("session-001")

        s2 = logger.create_session("session-002", "user-bob", "Bob")
        logger.log_message("session-002", MessageRole.USER, "我的耳机有杂音！")
        logger.log_message("session-002", MessageRole.AGENT, "抱歉给您带来不便，请问是什么型号的耳机？", emotion="frustrated")
        logger.log_message("session-002", MessageRole.USER, "StarPods Pro，买了两周了")
        logger.log_message("session-002", MessageRole.AGENT, "已为您创建退货工单 TK-20240120-00001。")
        s2.ticket_id = "TK-20240120-00001"
        logger.close_session("session-002")

        s3 = logger.create_session("session-003", "user-charlie", "Charlie")
        logger.log_message("session-003", MessageRole.USER, "你们什么破客服！问了三遍了！我要投诉！")
        logger.log_message("session-003", MessageRole.AGENT, "非常抱歉！正在为您转接人工客服...", emotion="angry")
        logger.close_session("session-003", status="escalated")

        # --- 查询与统计 ---
        print("--- 会话列表 ---\n")
        for session in logger.list_sessions():
            summary = session.to_summary()
            print(f"  [{summary['status']:10s}] {summary['session_id']} | "
                  f"{summary['user_name']:8s} | {summary['total_messages']} 条消息 | "
                  f"{summary['first_message'][:30]}...")

        print(f"\n--- 统计数据 ---\n")
        stats = logger.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # --- 导出会话记录 ---
        print(f"\n--- 导出会话记录（文本格式）---\n")
        text_export = logger.export_session("session-002", format="text")
        print(text_export)

    print("\n✓ 第六章示例 1 运行完成！")
    print("  下一课：02_review_analyzer.py - 对话复盘分析")


if __name__ == "__main__":
    main()
