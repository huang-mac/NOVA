# NOVA — 从零到企业级：智能客服工单 Agent 进阶之路

> 个人学习项目：每一天都在逼近一个完整的企业级 AI Agent 应用。

这不是一个"演示项目"，而是一条**从 LLM 第一行代码到可上线智能客服系统**的完整学习轨迹。每一章解决一个真实问题，每一章都在上一章的基础上长出新的能力。

---

## 学习路线图：从"会说话"到"能办事"

```
第一章          第二章           第三章           第四章
LLM 基础  →   对话记忆    →   RAG 知识库  →   情绪路由
"会说话"      "记得住"        "懂业务"        "有情商"
  │              │               │               │
  └──────────────┴───────────────┴───────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    第五章           第六章           第七章
   工单系统        审查监控        Function Call
  "能处理"        "可度量"        "会动手"
          │               │               │
          └───────────────┼───────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    第八章           第九章           第十章
 Plan-Execute     Multi-Agent      MCP 技能
 "会规划"        "能协作"        "可扩展"
          │               │               │
          └───────────────┴───────────────┘
                          ↓
                    第十一章
                  完整系统集成
                 "企业级上线"
```

---

## 逐章进阶详解

### 第一章 · LLM 基础 → 「让 AI 开口说话」

| 文件 | 学到什么 |
|------|----------|
| `01_hello_llm.py` | 第一次调用大模型——理解 HumanMessage / SystemMessage 的区别 |
| `02_system_prompt.py` | 系统提示词工程——设定角色、约束输出格式、注入业务规则 |
| `03_basic_chatbot.py` | 组合成一个可交互的命令行聊天机器人 |

> **此时的能力：** 一个能聊天的 Bot。但它没有记忆，每次对话都是"初次见面"。

---

### 第二章 · 对话记忆 → 「记住上一轮说了什么」

从这章开始，Bot 不再"金鱼记忆"。依次实现三种记忆策略：

| 文件 | 学到什么 | 适用场景 |
|------|----------|----------|
| `01_buffer_memory.py` | 全历史缓冲——把所有对话塞进上下文 | 短对话 |
| `02_summary_memory.py` | AI 摘要记忆——让 LLM 自动压缩历史 | 长对话，省 Token |
| `03_sliding_window.py` | 滑动窗口——只保留最近 N 轮 | 实时性优先 |
| `04_multi_turn_bot.py` | **融合三种策略**，实现多轮对话机器人 | 生产推荐 |

> **此时的能力：** Bot 能记住上下文了。但它还是"通用聊天"，不懂业务知识。

---

### 第三章 · RAG 知识库 → 「让 AI 读懂公司文档」

这是从"通用 AI"到"业务 AI"的关键一跃。

| 文件 | 学到什么 |
|------|----------|
| `01_document_loader.py` | 加载 Markdown 文档、文本分块策略（Chunk Size / Overlap） |
| `02_vector_store.py` | 搭建 Milvus 向量数据库、文本 Embedding、相似度检索 |
| `03_rag_qa.py` | 检索增强生成（RAG）——先搜后答，回答有理有据 |
| `04_full_rag_bot.py` | 带记忆 + 带知识库的完整客服机器人 |

> **此时的能力：** Bot 能根据公司产品手册回答专业问题。但它对所有用户"一视同仁"——不懂情绪。

---

### 第四章 · 情绪识别与智能路由 → 「察言观色，见人下菜碟」

| 文件 | 学到什么 |
|------|----------|
| `01_emotion_detector.py` | 用 Pydantic 结构化输出做情绪检测（开心/困惑/愤怒/焦虑） |
| `02_intent_classifier.py` | 意图分类（售后服务 / 技术咨询 / 投诉 / 购买咨询） |
| `03_smart_router.py` | **情绪 × 意图 = 路由决策**——紧急 + 愤怒 → 转人工；普通 + 咨询 → RAG 回复 |

> **此时的能力：** Bot 有了"情商"——对愤怒用户安抚并优先处理，对普通用户高效应答。但它还不会记录和追踪问题。

---

### 第五章 · 工单系统 → 「把问题管起来」

从"一问一答"升级到"全生命周期管理"。

| 文件 | 学到什么 |
|------|----------|
| `01_ticket_system.py` | 工单 CRUD — Pydantic 数据模型、JSON 持久化、自动编号 |
| `02_escalation_flow.py` | **升级状态机**：待处理 → 处理中 → 已解决 → 已关闭，带 SLA 超时 |
| `03_full_service.py` | 客服 + 工单的完整服务——自动创建工单、跟踪进展 |

> **此时的能力：** 问题被结构化记录、可追踪、可升级。但缺少对服务质量的度量。

---

### 第六章 · 审查与监控 → 「服务质量可量化」

| 文件 | 学到什么 |
|------|----------|
| `01_chat_logger.py` | 聊天记录存储——按会话/用户/时间三维查询 |
| `02_review_analyzer.py` | AI 驱动的对话复盘——找亮点、找问题、提改进建议 |
| `03_quality_scorer.py` | 多维度质量评分（准确性/礼貌度/效率/解决率） |
| `04_dashboard.py` | Streamlit 可视化监控面板 |

> **此时的能力：** 服务质量从"感觉还行"变成"数据说话"。但 Agent 还只能回答问题，不能执行操作。

---

### 第七章 · Function Call → 「从动嘴到动手」

这是 Agent 进化的分水岭——AI 不再只是"说话"，而是真正"做事"。

| 文件 | 学到什么 |
|------|----------|
| `01_tool_basics.py` | `@tool` 装饰器 → JSON Schema 自动生成 → Function Call 双步调用 |
| `02_customer_tools.py` | 模拟真实业务工具：查订单、查物流、查库存、发起退款 |
| `03_tool_prompt.py` | 工具场景下的 Prompt 工程——何时调用工具、何时直接回复 |
| `04_multi_tool_agent.py` | 多工具编排——LLM 自主决策用哪个工具、按什么顺序 |

> **此时的能力：** Agent 能查询数据库、执行退款。但遇到复杂问题（"先查订单、再查物流、最后决定是否补发"），需要多步推理。

---

### 第八章 · Plan-Execute → 「三思而后行」

引入 LangGraph，让 Agent 先规划再执行。

| 文件 | 学到什么 |
|------|----------|
| `01_graph_basics.py` | LangGraph 核心概念——StateGraph、Node、Edge、条件分支 |
| `02_planner.py` | LLM 把用户复杂问题拆解为一步步可执行的计划 |
| `03_executor.py` | 按计划顺序执行——每步调用工具，收集结果 |
| `04_replanner.py` | **关键突破**：执行中发现问题 → 自动调整计划（Replan） |
| `05_full_planner_agent.py` | 完整 Plan → Execute → Replan 循环 |

> **此时的能力：** Agent 能处理需要多步推理的复杂问题，遇到意外会自动调整策略。但它是"单打独斗"。

---

### 第九章 · Multi-Agent → 「让多个 Agent 分工协作」

一个 Agent 不够，需要一支团队。

| 文件 | 学到什么 |
|------|----------|
| `01_reception_agent.py` | 前台接待 Agent——接客、识别意图、分流 |
| `02_specialized_agents.py` | 三个专业 Agent：售后专员、技术支持、值班主管 |
| `03_agent_communication.py` | Agent 间通信协议——消息传递、任务委派、结果汇总 |
| `04_multi_agent_system.py` | **4 个 Agent 协同工作**——接待 → 路由 → 专业 Agent → 回复 |

> **此时的能力：** 不同类型的用户请求由不同 Agent 处理——售后找售后专员，技术问题找技术专家，投诉自动升级给主管。

---

### 第十章 · MCP 技能 → 「插件化扩展」

MCP（Model Context Protocol）让 Agent 的能力可以无限扩展。

| 文件 | 学到什么 |
|------|----------|
| `01_mcp_server.py` | 实现简化版 MCP Server——注册技能、暴露接口 |
| `02_mcp_client.py` | MCP Client——连接多个 Server、聚合所有可用技能 |
| `03_skills.py` | 技能定义（天气查询、计算器、搜索、翻译…） |
| `04_skill_manager.py` | 技能管理器：注册/列表/调用/热加载 |
| `05_mcp_skills_integration.py` | **Agent × MCP**：Agent 动态发现并调用外部技能 |

> **此时的能力：** 不需要改 Agent 代码就能加新功能——新增一个 MCP Server，Agent 自动获得新技能。

---

### 第十一章 · 完整系统集成 → 「一切就绪，准备上线」

所有前置章节的能力在此汇聚，成为一套可部署的系统。

| 文件 | 职责 |
|------|------|
| `agent_core.py` | **核心引擎**：情绪检测 → 意图分类 → 路由分发 → RAG/工单/转人工 → 日志记录 |
| `api_server.py` | FastAPI REST 服务：`/api/chat`、`/api/health`、`/api/tickets` |
| `web_ui.py` | Streamlit Web 聊天界面——用户最终看到的交互层 |

```
                   ┌──────────────────────┐
                   │   Streamlit Web UI   │  ← 用户浏览器
                   └──────────┬───────────┘
                              │ HTTP
                   ┌──────────▼───────────┐
                   │   FastAPI Server     │  ← REST API 层
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │    Agent Core        │  ← 核心编排引擎
                   │  情绪 → 意图 → 路由   │
                   └──┬──────┬──────┬─────┘
                      │      │      │
              ┌───────▼┐ ┌──▼───┐ ┌▼───────┐
              │  RAG   │ │工单  │ │转人工  │  ← 执行层
              └───────┘ └──────┘ └────────┘
```

> **此时的能力：** 一个完整的企业级智能客服系统——有前端、有后端、有知识库、有工单、有监控、可扩展。

---

## 项目结构

```
NOVA/
├── ch01/                          # 第一章：LLM 基础
├── ch02-memory/                   # 第二章：对话记忆
├── ch03-rag/                      # 第三章：RAG 知识库
├── ch04-emotion-routing/          # 第四章：情绪识别与路由
├── ch05-ticket-escalation/        # 第五章：工单系统
├── ch06-review-monitor/           # 第六章：审查与监控
├── ch07-function-call/            # 第七章：Function Call
├── ch08-plan-execute/             # 第八章：Plan-Execute (LangGraph)
├── ch09-multi-agent/              # 第九章：Multi-Agent
├── ch10-mcp-skills/               # 第十章：MCP 技能扩展
├── ch11-full-system/              # 第十一章：完整系统集成
│   ├── agent_core.py              #   核心 Agent 引擎
│   ├── api_server.py              #   FastAPI REST 服务
│   └── web_ui.py                  #   Streamlit Web 界面
├── knowledge_base/                # 知识库文档（用于 RAG）
├── config.py                      # 共享配置（LLM / Embeddings / Milvus）
├── requirements.txt               # 依赖清单
└── .env.example                   # 环境变量模板
```

## 快速开始

```bash
# 1. 环境准备
python -m venv venv
source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt

# 2. 配置密钥
cp .env.example .env          # 编辑 .env 填入 API Key

# 3. 按章节学习（每个脚本独立可运行）
python ch01/01_hello_llm.py           # 第一个 LLM 调用
python ch03-rag/04_full_rag_bot.py    # RAG 知识库问答  
python ch08-plan-execute/05_full_planner_agent.py  # Plan-Execute Agent

# 4. 启动完整系统
python ch11-full-system/api_server.py         # 先启动 API
streamlit run ch11-full-system/web_ui.py      # 再启动 Web 界面
```

## 技术栈

| 层 | 技术选型 | 为什么选它 |
|----|----------|------------|
| LLM 框架 | LangChain ≥ 1.2 | 最成熟的 LLM 应用框架，生态完善 |
| Agent 编排 | LangGraph ≥ 1.1 | 有向图编排，比 Chain 更灵活 |
| 向量数据库 | Milvus / FAISS | 支持本地 Lite 模式（零部署）和生产集群模式 |
| REST API | FastAPI | 高性能异步、自动生成 OpenAPI 文档 |
| Web 界面 | Streamlit | 纯 Python 写前端，最快出原型 |
| 数据校验 | Pydantic ≥ 2.0 | 结构化输出，LLM 输出可校验 |

## License

MIT
