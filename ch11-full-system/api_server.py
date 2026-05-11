"""
===============================================
第十一章：FastAPI 服务接口
===============================================
目标：将 Agent 包装为 REST API 服务，支持 HTTP 调用。

API 接口：
- POST /api/chat          - 发送消息并获取回复
- GET  /api/session/{id}  - 获取会话信息
- GET  /api/tickets       - 查询工单列表
- GET  /api/health        - 健康检查

运行方式：
    cd ch11-full-system
    python api_server.py

    服务启动后访问：
    http://localhost:8000/docs   (Swagger API 文档)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# 导入核心 Agent
from agent_core import CustomerServiceAgent, AgentResponse


# ============================================================
# API 数据模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求。"""
    session_id: str = Field(description="会话 ID（前端生成或后端分配）")
    user_message: str = Field(description="用户消息")
    user_id: str = Field(default="anonymous", description="用户 ID")


class ChatResponse(BaseModel):
    """聊天响应。"""
    reply: str
    session_id: str
    emotion: Optional[str] = None
    intent: Optional[str] = None
    action: Optional[str] = None
    ticket_id: Optional[str] = None
    escalation_id: Optional[str] = None
    confidence: float = 0
    processing_time_ms: float = 0


class SessionInfo(BaseModel):
    """会话信息。"""
    session_id: str
    message_count: int
    emotions: dict
    actions: dict
    has_ticket: bool
    has_escalation: bool


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str
    uptime: str
    version: str


# ============================================================
# 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="星辰科技 · 智能客服 API",
    description="基于 LangChain 的智能客服工单 Agent 系统",
    version="1.0.0",
)

# 允许跨域（Web UI 需要调用 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Agent（全局单例）
agent = CustomerServiceAgent(use_llm=False, use_rag=False)
_start_time = datetime.now()


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口。"""
    uptime = str(datetime.now() - _start_time).split(".")[0]
    return HealthResponse(
        status="healthy",
        uptime=uptime,
        version="1.0.0",
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """
    发送消息并获取 Agent 回复。

    这是核心接口，完整的处理流程：
    用户消息 → 情绪检测 → 意图分类 → 智能路由 → 执行动作 → 返回响应
    """
    try:
        response = agent.chat(
            session_id=request.session_id,
            user_message=request.user_message,
            user_id=request.user_id,
        )

        return ChatResponse(
            reply=response.reply,
            session_id=response.session_id,
            emotion=response.emotion.value if response.emotion else None,
            intent=response.intent.value if response.intent else None,
            action=response.action.value if response.action else None,
            ticket_id=response.ticket_id,
            escalation_id=response.escalation_id,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}", response_model=SessionInfo, tags=["会话"])
async def get_session(session_id: str):
    """获取会话摘要信息。"""
    summary = agent.get_session_summary(session_id)
    if summary["message_count"] == 0:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return SessionInfo(**summary)


@app.get("/api/session/{session_id}/log", tags=["会话"])
async def get_session_log(session_id: str):
    """获取会话的完整聊天记录。"""
    log = agent.get_session_log(session_id)
    if not log:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return {"session_id": session_id, "messages": log}


@app.get("/api/tickets", tags=["工单"])
async def list_tickets(user_id: Optional[str] = None):
    """查询工单列表。"""
    if user_id:
        tickets = agent.ticket_manager.list_by_user(user_id)
    else:
        tickets = list(agent.ticket_manager._tickets.values())
    return {"tickets": tickets, "total": len(tickets)}


@app.get("/api/tickets/{ticket_id}", tags=["工单"])
async def get_ticket(ticket_id: str):
    """查询单个工单详情。"""
    ticket = agent.ticket_manager.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")
    return ticket


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  星辰科技 · 智能客服 API 服务")
    print("=" * 60)
    print(f"  服务地址: http://localhost:8000")
    print(f"  API 文档: http://localhost:8000/docs")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
