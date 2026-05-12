"""
===============================================
第二章 示例 4：完整多轮客服系统
===============================================
目标：整合记忆系统，实现一个支持多轮对话、多会话管理的完整客服。

知识点：
    1. 手动实现滑动窗口记忆
    2. 多用户会话管理（不同用户独立记忆）
    3. 会话超时自动清理
    4. 对话导出功能

注意：LangChain 1.x 已移除 ConversationBufferWindowMemory，
      改为手动维护消息列表并裁切。

运行方式：
    cd ch02-memory
    python 04_multi_turn_bot.py
"""

import sys
import os
import json
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm_config


# ============================================================
# 客服系统提示词
# ============================================================
SYSTEM_PROMPT = """你是「星辰科技」的智能客服小星。

## 核心能力
- 记住用户在本次对话中提供的所有信息
- 根据对话历史提供连贯的回复
- 主动关联之前提到的信息

## 对话风格
- 简洁明了，不啰嗦
- 如果用户提到之前的信息，先确认再回答
- 使用用户的名字增加亲切感

## 产品信息
- StarPods Pro：旗舰降噪耳机，¥799（活动价），30h续航，IPX4防水
- StarWatch X：智能手表，¥1299，心率血氧监测，7天续航
- StarBox：智能音箱，¥399，Hi-Fi音质，全屋智能控制

## 售后政策
- 7天无理由退换
- 1年官方保修
- 配送：顺丰 1-3天
- 售后热线：400-888-9999
"""


# ============================================================
# 会话管理器
# ============================================================
class SessionManager:
    """
    管理多个用户的会话记忆。

    功能：
    - 为每个 session_id 维护独立的对话记忆（滑动窗口）
    - 自动清理超时会话（默认30分钟）
    - 导出会话记录
    """

    def __init__(self, window_size: int = 5, timeout_minutes: int = 30):
        """
        Args:
            window_size: 滑动窗口大小（保留最近几轮对话）
            timeout_minutes: 会话超时时间（分钟）
        """
        self.window_size = window_size
        self.timeout_minutes = timeout_minutes
        # {session_id: {"messages": [...], "last_active": datetime, ...}}
        self._sessions = {}

    def get_memory(self, session_id: str) -> list:
        """获取或创建指定 session 的消息列表（滑动窗口）。"""
        now = datetime.now()

        # 清理超时会话
        self._cleanup_expired(now)

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "last_active": now,
                "created_at": now,
                "message_count": 0,
            }

        self._sessions[session_id]["last_active"] = now
        return self._sessions[session_id]["messages"]

    def save_context(self, session_id: str, user_input: str, ai_output: str):
        """保存对话并裁剪到窗口大小。"""
        messages = self.get_memory(session_id)
        messages.append(HumanMessage(content=user_input))
        messages.append(AIMessage(content=ai_output))
        # 裁剪到 window_size 轮（每轮 2 条）
        max_size = self.window_size * 2
        if len(messages) > max_size:
            self._sessions[session_id]["messages"] = messages[-max_size:]
        self._sessions[session_id]["message_count"] += 2

    def _cleanup_expired(self, now: datetime):
        """清理超时的会话。"""
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session["last_active"] > timedelta(minutes=self.timeout_minutes)
        ]
        for sid in expired:
            del self._sessions[sid]
            print(f"  [系统] 会话 {sid} 已超时清理")

    def get_session_info(self, session_id: str) -> dict:
        """获取会话信息。"""
        if session_id not in self._sessions:
            return {"status": "not_found"}
        session = self._sessions[session_id]
        return {
            "status": "active",
            "created_at": session["created_at"].isoformat(),
            "last_active": session["last_active"].isoformat(),
            "message_count": session["message_count"],
        }

    def export_session(self, session_id: str) -> str:
        """导出会话记录为 JSON 字符串。"""
        if session_id not in self._sessions:
            return "{}"
        messages = self._sessions[session_id]["messages"]
        records = []
        for msg in messages:
            records.append({
                "role": msg.type,
                "content": msg.content,
            })
        return json.dumps(records, ensure_ascii=False, indent=2)

    def list_sessions(self) -> list:
        """列出所有活跃会话。"""
        return [
            {"session_id": sid, **self.get_session_info(sid)}
            for sid in self._sessions
        ]


# ============================================================
# 完整多轮客服机器人
# ============================================================
class MultiTurnCustomerServiceBot:
    """
    完整的多轮客服机器人。

    相比第一章的基础版，增加了：
    - 滑动窗口记忆（记住最近 N 轮对话）
    - 多会话管理（不同用户独立对话）
    - 会话超时清理
    - 会话记录导出
    """

    def __init__(self, window_size: int = 5):
        config = get_llm_config()
        self.llm = ChatOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model_name"],
            temperature=0.5,
        )
        self.session_manager = SessionManager(window_size=window_size)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),
        ])
        self.chain = self.prompt | self.llm

    def chat(self, session_id: str, user_input: str) -> str:
        """
        处理用户消息并返回回复。

        Args:
            session_id: 会话标识（通常是用户ID或浏览器 session）
            user_input: 用户输入
        """
        if user_input.strip().lower() in ("退出", "quit", "exit"):
            return "感谢使用星辰科技客服，再见！"

        # 获取该会话的记忆
        messages = self.session_manager.get_memory(session_id)

        # 调用 LLM
        response = self.chain.invoke({
            "chat_history": messages,
            "user_input": user_input,
        })

        # 保存到记忆（自动裁剪到窗口大小）
        self.session_manager.save_context(session_id, user_input, response.content)

        return response.content


# ============================================================
# 演示模式：模拟多用户对话
# ============================================================
def demo_multi_user():
    """演示多用户同时对话的场景。"""
    print("\n--- 演示：多用户同时对话 ---\n")

    bot = MultiTurnCustomerServiceBot(window_size=5)

    # 模拟三个用户的对话交叉进行
    sessions = {
        "user-alice": [
            "你好，我叫 Alice",
            "我想买一副蓝牙耳机",
            "StarPods Pro 多少钱？",
            "续航怎么样？",
        ],
        "user-bob": [
            "Hi，我的手表屏幕不亮了",
            "型号是 StarWatch X",
            "还在保修期内吗？买了3个月了",
        ],
        "user-charlie": [
            "你们的音箱 StarBox 支持哪些语音助手？",
            "能控制空调吗？",
        ],
    }

    for i in range(max(len(v) for v in sessions.values())):
        print(f"\n{'='*50}")
        print(f"  [时间步 {i+1}]")
        print(f"{'='*50}")

        for session_id, messages_list in sessions.items():
            if i < len(messages_list):
                user_input = messages_list[i]
                user_name = session_id.split("-")[1].title()

                response = bot.chat(session_id, user_input)
                print(f"\n👤 {user_name}: {user_input}")
                print(f"🤖 小星: {response}")

    # 展示会话信息
    print(f"\n\n--- 会话统计 ---")
    for session_id in sessions:
        info = bot.session_manager.get_session_info(session_id)
        print(f"  {session_id}: {info['message_count']} 条消息, "
              f"最后活跃: {info['last_active'][:19]}")


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 60)
    print("  第二章 示例 4：完整多轮客服系统")
    print("=" * 60)

    # 演示多用户对话
    demo_multi_user()

    # 导出会话记录示例
    print("\n\n--- 导出 Alice 的会话记录 ---")
    bot = MultiTurnCustomerServiceBot()
    for msg in ["我叫 Alice", "StarPods Pro 怎么样？"]:
        bot.chat("export-demo", msg)

    exported = bot.session_manager.export_session("export-demo")
    print(exported)

    print("\n✓ 第二章示例 4 运行完成！")
    print("  下一章：ch03-rag - 知识库检索 RAG")


if __name__ == "__main__":
    main()
