"""
===============================================
第十一章：Streamlit Web 交互界面
===============================================
目标：为智能客服系统提供友好的 Web 聊天界面。

运行方式：
    cd ch11-full-system
    streamlit run web_ui.py

    需要先安装 streamlit: pip install streamlit
    需要先启动 API 服务: python api_server.py
"""

import streamlit as st
import requests
import uuid
from datetime import datetime

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="星辰科技 · 智能客服",
    page_icon="🌟",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义样式
# ============================================================

st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        max-width: 80%;
        word-wrap: break-word;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .agent-message {
        background: #f0f2f5;
        color: #333;
        margin-right: auto;
    }
    .system-message {
        background: #fff3cd;
        color: #856404;
        text-align: center;
        font-size: 12px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-left: 8px;
    }
    .ticket-info {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# API 客户端
# ============================================================

API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")


def send_message(session_id: str, user_message: str, user_id: str = "web-user") -> dict:
    """发送消息到 API。"""
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "session_id": session_id,
                "user_message": user_message,
                "user_id": user_id,
            },
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"reply": "⚠️ 无法连接到服务端，请确保 API 服务已启动（python api_server.py）"}
    except Exception as e:
        return {"reply": f"⚠️ 服务异常: {str(e)}"}


def get_session_info(session_id: str) -> dict:
    """获取会话信息。"""
    try:
        response = requests.get(f"{API_BASE}/api/session/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


# ============================================================
# 情绪 emoji 映射
# ============================================================

EMOTION_EMOJI = {
    "positive": "😊", "neutral": "😐", "confused": "🤔",
    "frustrated": "😤", "angry": "😡", "anxious": "😰", "sad": "😢",
}

ACTION_LABEL = {
    "greeting": "👋 问候", "rag_query": "🔍 知识库",
    "create_ticket": "📋 工单", "escalate": "🚨 转人工",
    "llm_fallback": "🤖 智能回答",
}


# ============================================================
# 主界面
# ============================================================

def main():
    # --- 侧边栏 ---
    with st.sidebar:
        st.title("🌟 星辰科技")
        st.caption("智能客服工单 Agent")

        st.divider()

        # API 配置
        api_url = st.text_input("API 地址", value=API_BASE, key="api_url")

        # 会话管理
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]

        st.session_state.session_id = st.text_input(
            "会话 ID",
            value=st.session_state.session_id,
            key="session_input",
        )

        if st.button("🔄 新会话", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # 系统信息
        st.subheader("📊 系统信息")
        try:
            health = requests.get(f"{api_url}/api/health", timeout=3).json()
            st.success(f"✅ 服务正常")
            st.caption(f"运行时间: {health.get('uptime', 'N/A')}")
            st.caption(f"版本: {health.get('version', 'N/A')}")
        except Exception:
            st.error("❌ 服务未连接")

        st.divider()
        st.caption("基于 LangChain 构建")
        st.caption("© 2024 星辰科技")

    # --- 主内容区 ---
    st.title("💬 智能客服对话")

    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 显示元数据
            if msg.get("metadata"):
                meta = msg["metadata"]
                parts = []
                if meta.get("emotion"):
                    emoji = EMOTION_EMOJI.get(meta["emotion"], "")
                    parts.append(f"情绪: {emoji} {meta['emotion']}")
                if meta.get("action"):
                    parts.append(f"动作: {ACTION_LABEL.get(meta['action'], meta['action'])}")
                if meta.get("processing_time"):
                    parts.append(f"耗时: {meta['processing_time']}ms")
                if parts:
                    st.caption(" | ".join(parts))

            # 显示工单信息
            if msg.get("ticket_id"):
                st.info(f"📋 工单号: {msg['ticket_id']} — 已创建，客服将尽快联系您")
            if msg.get("escalation_id"):
                st.warning(f"🚨 转人工请求: {msg['escalation_id']} — 正在为您转接")

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用 API
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = send_message(
                    st.session_state.session_id,
                    prompt,
                )

            st.markdown(result["reply"])

            # 元数据
            meta = {}
            if result.get("emotion"):
                meta["emotion"] = result["emotion"]
            if result.get("action"):
                meta["action"] = result["action"]
            if result.get("processing_time_ms"):
                meta["processing_time"] = result["processing_time_ms"]

            if meta:
                parts = []
                if meta.get("emotion"):
                    emoji = EMOTION_EMOJI.get(meta["emotion"], "")
                    parts.append(f"情绪: {emoji} {meta['emotion']}")
                if meta.get("action"):
                    parts.append(f"动作: {ACTION_LABEL.get(meta['action'], meta['action'])}")
                if meta.get("processing_time"):
                    parts.append(f"耗时: {meta['processing_time']}ms")
                st.caption(" | ".join(parts))

            # 工单/转人工提示
            msg_data = {"role": "assistant", "content": result["reply"], "metadata": meta}
            if result.get("ticket_id"):
                msg_data["ticket_id"] = result["ticket_id"]
                st.info(f"📋 工单号: {result['ticket_id']} — 已创建")
            if result.get("escalation_id"):
                msg_data["escalation_id"] = result["escalation_id"]
                st.warning(f"🚨 转人工: {result['escalation_id']} — 转接中")

            st.session_state.messages.append(msg_data)


if __name__ == "__main__":
    main()
