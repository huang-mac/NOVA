"""
===============================================
第十一章 核心：智能客服 Agent 完整实现
===============================================
目标：将前十一章所有模块整合为一个完整的智能客服 Agent。

这是整个系列课程的核心文件，整合了：
- 第一章：LLM 调用与系统提示词
- 第二章：多轮对话记忆
- 第三章：RAG 知识库检索
- 第四章：情绪识别与智能路由
- 第五章：工单系统与转人工
- 第六章：聊天记录与复盘

设计原则：
1. 模块化：每个组件独立，可单独替换
2. 流水线：用户消息经过完整的处理链路
3. 可扩展：方便添加新功能
4. 可观测：完整的日志和统计

运行方式：
    cd ch11-full-system
    python agent_core.py
"""

import os
import sys
import json
import uuid
import tempfile
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# 确保能导入 config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 第一部分：数据模型（统一的数据定义）
# ============================================================

class EmotionType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    SAD = "sad"


class IntentType(str, Enum):
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    PRICE_INQUIRY = "price_inquiry"
    REFUND = "refund"
    RETURN = "return"
    EXCHANGE = "exchange"
    WARRANTY = "warranty"
    REPAIR = "repair"
    DELIVERY = "delivery"
    TROUBLESHOOT = "troubleshoot"
    USAGE_GUIDE = "usage_guide"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"


class RouteAction(str, Enum):
    GREETING = "greeting"
    RAG_QUERY = "rag_query"
    CREATE_TICKET = "create_ticket"
    ESCALATE = "escalate"
    LLM_FALLBACK = "llm_fallback"


class ChatMessage(BaseModel):
    """单条消息。"""
    role: str  # user / agent / system
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Agent 的统一响应格式。"""
    reply: str                                    # 给用户的回复
    session_id: str                               # 会话 ID
    emotion: Optional[EmotionType] = None         # 检测到的情绪
    intent: Optional[IntentType] = None           # 识别的意图
    action: Optional[RouteAction] = None          # 执行的动作
    ticket_id: Optional[str] = None               # 创建的工单 ID
    escalation_id: Optional[str] = None           # 转人工请求 ID
    confidence: float = 0.0                       # 置信度
    processing_time_ms: float = 0                 # 处理耗时（毫秒）


# ============================================================
# 第二部分：情绪检测模块
# ============================================================

class EmotionDetector:
    """
    用户情绪检测器。

    支持两种模式：
    1. LLM 模式：使用 LLM 进行精确分析（需 API 调用）
    2. 关键词模式：基于关键词匹配（快速，无 API 开销）
    """

    # 关键词模式用的情绪词典
    EMOTION_KEYWORDS = {
        EmotionType.ANGRY: ["垃圾", "破", "废物", "投诉", "滚", "骗子", "恶心", "什么破", "太过分"],
        EmotionType.FRUSTRATED: ["慢", "等了", "还没", "多次", "又坏了", "又不", "烦死了"],
        EmotionType.ANXIOUS: ["急", "快点", "赶紧", "什么时候", "催", "等着急"],
        EmotionType.SAD: ["失望", "难过", "伤心", "心寒", "再也不"],
        EmotionType.POSITIVE: ["谢谢", "满意", "不错", "好用", "很好", "点赞"],
        EmotionType.CONFUSED: ["怎么", "看不懂", "不会用", "搞不懂", "什么意思"],
    }

    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: 是否使用 LLM 进行情绪分析（默认用关键词）
        """
        self.use_llm = use_llm
        self._llm = None

    def detect(self, message: str) -> dict:
        """
        检测用户消息中的情绪。

        Returns:
            dict: {"emotion": EmotionType, "intensity": int, "should_escalate": bool}
        """
        if self.use_llm:
            return self._detect_with_llm(message)
        return self._detect_with_keywords(message)

    def _detect_with_keywords(self, message: str) -> dict:
        """基于关键词的快速情绪检测。"""
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            scores[emotion] = sum(1 for kw in keywords if kw in message)

        if not any(scores.values()):
            return {"emotion": EmotionType.NEUTRAL, "intensity": 2, "should_escalate": False}

        top_emotion = max(scores, key=scores.get)
        intensity = min(10, scores[top_emotion] * 3 + 2)
        should_escalate = top_emotion == EmotionType.ANGRY and intensity >= 8

        return {"emotion": top_emotion, "intensity": intensity, "should_escalate": should_escalate}

    def _detect_with_llm(self, message: str) -> dict:
        """使用 LLM 进行精确情绪分析。"""
        try:
            from config import get_llm
            llm = get_llm(temperature=0.1)
            response = llm.invoke(f"""分析以下消息的情绪，返回JSON格式：
{{"emotion": "positive/neutral/confused/frustrated/angry/anxious/sad", "intensity": 1-10, "should_escalate": true/false}}

消息：「{message}」""")

            return json.loads(response.content)
        except Exception as e:
            # LLM 失败时降级到关键词模式
            print(f"  [警告] LLM 情绪分析失败: {e}，降级到关键词模式")
            return self._detect_with_keywords(message)


# ============================================================
# 第三部分：意图分类模块
# ============================================================

class IntentClassifier:
    """
    用户意图分类器。

    将用户消息分类为预定义的意图类别。
    """

    INTENT_KEYWORDS = {
        IntentType.GREETING: ["你好", "在吗", "hello", "hi", "嗨", "hey"],
        IntentType.PRODUCT_INQUIRY: ["什么产品", "有哪些", "推荐", "参数", "功能", "支持什么"],
        IntentType.PRICE_INQUIRY: ["多少钱", "价格", "优惠", "折扣", "促销", "活动"],
        IntentType.REFUND: ["退款"],
        IntentType.RETURN: ["退货"],
        IntentType.EXCHANGE: ["换货", "换新", "更换"],
        IntentType.WARRANTY: ["保修", "质保", "保修期"],
        IntentType.REPAIR: ["维修", "修", "坏了"],
        IntentType.DELIVERY: ["发货", "快递", "配送", "物流", "到货"],
        IntentType.TROUBLESHOOT: ["故障", "有问题", "不正常", "连不上", "没反应", "杂音"],
        IntentType.USAGE_GUIDE: ["怎么用", "怎么操作", "使用方法", "如何", "教程"],
        IntentType.COMPLAINT: ["投诉", "不满", "差评", "举报"],
        IntentType.ESCALATION: ["转人工", "人工客服", "真人", "活人"],
    }

    def classify(self, message: str) -> dict:
        """
        分类用户消息的意图。

        Returns:
            dict: {"primary_intent": IntentType, "confidence": float, "entities": dict}
        """
        message_lower = message.lower()

        for intent, keywords in self.INTENT_KEYWORDS.items():
            if any(kw in message_lower for kw in keywords):
                return {
                    "primary_intent": intent,
                    "confidence": 0.8,
                    "entities": {},
                }

        return {"primary_intent": IntentType.UNKNOWN, "confidence": 0.3, "entities": {}}


# ============================================================
# 第四部分：智能路由模块
# ============================================================

class SmartRouter:
    """
    智能路由器：根据情绪和意图决定处理路径。
    """

    def route(self, emotion: dict, intent: dict) -> dict:
        """
        路由决策。

        Returns:
            dict: {"action": RouteAction, "priority": str, "emotion_prefix": str}
        """
        should_escalate = emotion.get("should_escalate", False)
        emotion_type = emotion.get("emotion", EmotionType.NEUTRAL)
        intensity = emotion.get("intensity", 2)
        primary_intent = intent.get("primary_intent", IntentType.UNKNOWN)

        # 优先级 1：紧急情绪 / 明确要求转人工
        if should_escalate or primary_intent in (IntentType.ESCALATION, IntentType.COMPLAINT):
            return {
                "action": RouteAction.ESCALATE,
                "priority": "urgent" if should_escalate else "high",
                "emotion_prefix": "非常抱歉给您带来不好的体验！" if should_escalate else "好的，马上为您转接人工客服。",
            }

        # 优先级 2：售后工单类
        ticket_intents = {IntentType.REFUND, IntentType.RETURN, IntentType.EXCHANGE,
                         IntentType.REPAIR, IntentType.FEEDBACK}
        if primary_intent in ticket_intents:
            prefix = "抱歉给您带来不便，" if intensity >= 6 else ""
            return {"action": RouteAction.CREATE_TICKET, "priority": "high" if intensity >= 6 else "normal", "emotion_prefix": prefix}

        # 优先级 3：咨询类
        rag_intents = {IntentType.PRODUCT_INQUIRY, IntentType.PRICE_INQUIRY,
                      IntentType.DELIVERY, IntentType.TROUBLESHOOT,
                      IntentType.USAGE_GUIDE, IntentType.WARRANTY}
        if primary_intent in rag_intents:
            return {"action": RouteAction.RAG_QUERY, "priority": "normal", "emotion_prefix": ""}

        # 优先级 4：问候
        if primary_intent == IntentType.GREETING:
            return {"action": RouteAction.GREETING, "priority": "normal", "emotion_prefix": ""}

        # 兜底
        return {"action": RouteAction.LLM_FALLBACK, "priority": "normal", "emotion_prefix": ""}


# ============================================================
# 第五部分：知识库 RAG 模块
# ============================================================

class RAGEngine:
    """
    RAG 知识库检索引擎。

    两种模式：
    1. 向量检索模式：使用 Milvus（需要运行 02_vector_store.py 构建索引）
    2. FAQ 匹配模式：基于关键词匹配（无需向量数据库）

    Milvus 说明：
    - 开发阶段用 Milvus Lite（文件模式，零部署）
    - 生产环境切换到 Milvus Server 或 Zilliz Cloud，改一下 URI 即可
    """

    # FAQ 知识库（简化版，实际应用中应从数据库/文件加载）
    FAQ = {
        "StarPods Pro": "StarPods Pro 是星辰科技旗舰降噪耳机，搭载 StarNoise 3.0 芯片，48dB 主动降噪，蓝牙 5.3，30h 续航，IPX4 防水，官方售价 ¥899（活动价 ¥799）。",
        "StarWatch X": "StarWatch X 是星辰科技健康智能手表，1.43寸 AMOLED 屏，心率血氧监测，GPS 定位，7 天续航，5ATM+IP68 防水，售价 ¥1299。",
        "StarBox": "StarBox 是星辰科技智能音箱，5 单元 Hi-Fi 音质，30W 功率，支持 StarVoice 语音助手，兼容 2000+ 品牌智能家居设备，售价 ¥399。",
        "退货": "7 天内无理由退货，商品需保持完好。请在「我的订单」中申请或联系客服处理。",
        "退款": "退款将在 3-5 个工作日内原路到账。",
        "保修": "所有产品提供 1 年官方保修，非人为损坏免费维修。维修周期 3-7 个工作日。",
        "发货": "默认顺丰包邮，下单后 1-3 个工作日送达。支持次日达（¥20）。",
        "售后热线": "售后热线：400-888-9999，人工服务时间：工作日 9:00-18:00。",
        "杂音": "耳机杂音处理：1. 删除蓝牙配对记录 2. 长按触控区 15 秒重置 3. 重新配对。如问题持续请申请售后。",
        "蓝牙": "蓝牙连接问题：1. 确保设备蓝牙已开启 2. 在手机蓝牙设置中忽略耳机 3. 放回充电盒重置后重新配对。",
        "屏幕不亮": "手表屏幕不亮处理：1. 长按侧边按钮 3 秒重启 2. 充电 15 分钟后重试 3. 用酒精棉擦拭充电触点。",
        "血氧": "在手表主界面点击血氧测量，保持静止佩戴 15 秒即可获取血氧饱和度数据。测量时请确保手表贴合手腕。",
        "智能控制": "StarBox 支持 2000+ 品牌智能设备，可通过语音或 App 控制。支持米家、HomeKit、天猫精灵、小度等协议。",
    }

    def __init__(self, use_vector_db: bool = False, milvus_uri: str = None):
        self.use_vector_db = use_vector_db
        self._retriever = None

        if use_vector_db and milvus_uri:
            try:
                from langchain_milvus import Milvus
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                from config import get_embeddings, get_milvus_config
                embeddings = get_embeddings()
                uri = milvus_uri or get_milvus_config()["uri"]
                self._vectorstore = Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": uri},
                    collection_name="customer_service_kb",
                )
                self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 3})
            except Exception as e:
                print(f"  [警告] 向量数据库加载失败: {e}，降级到 FAQ 模式")
                self.use_vector_db = False

    def query(self, question: str) -> str:
        """检索知识库并返回答案。"""
        if self.use_vector_db and self._retriever:
            return self._query_vector(question)
        return self._query_faq(question)

    def _query_vector(self, question: str) -> str:
        """使用向量数据库检索。"""
        docs = self._retriever.invoke(question)
        if not docs:
            return "抱歉，我没有找到相关信息。建议您拨打售后热线 400-888-9999。"
        # 取最相关的文档内容
        return docs[0].page_content

    def _query_faq(self, question: str) -> str:
        """基于 FAQ 关键词匹配检索。"""
        best_match = None
        best_score = 0

        for keyword, answer in self.FAQ.items():
            score = sum(1 for kw in keyword if kw in question)
            if score > best_score:
                best_score = score
                best_match = answer

        if best_match:
            return best_match
        return "抱歉，我暂时没有找到相关信息。建议您拨打售后热线 400-888-9999（工作日 9:00-18:00）。"


# ============================================================
# 第六部分：记忆管理模块
# ============================================================

class ConversationMemory:
    """
    对话记忆管理器。

    为每个 session_id 维护独立的对话历史。
    使用滑动窗口控制记忆大小。
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._sessions: dict[str, list[ChatMessage]] = {}

    def add_message(self, session_id: str, role: str, content: str, **metadata):
        """添加消息到会话。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        msg = ChatMessage(role=role, content=content, metadata=metadata)
        self._sessions[session_id].append(msg)

        # 滑动窗口裁剪
        if len(self._sessions[session_id]) > self.window_size * 2:
            self._sessions[session_id] = self._sessions[session_id][-self.window_size * 2:]

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """获取对话历史。"""
        return self._sessions.get(session_id, [])

    def get_history_text(self, session_id: str) -> str:
        """获取格式化的对话历史文本。"""
        messages = self.get_history(session_id)
        if not messages:
            return "（无历史对话）"
        lines = []
        for msg in messages:
            role = "用户" if msg.role == "user" else "客服小星"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear_session(self, session_id: str):
        """清除会话记忆。"""
        if session_id in self._sessions:
            del self._sessions[session_id]


# ============================================================
# 第七部分：工单模块（简化版）
# ============================================================

class SimpleTicketManager:
    """简化的工单管理器。"""

    def __init__(self):
        self._tickets: dict[str, dict] = {}
        self._counter = 0

    def create(self, category: str, title: str, description: str,
               user_id: str, priority: str = "normal", **kwargs) -> dict:
        """创建工单。"""
        self._counter += 1
        ticket_id = f"TK-{datetime.now().strftime('%Y%m%d')}-{self._counter:05d}"
        ticket = {
            "ticket_id": ticket_id,
            "category": category,
            "title": title,
            "description": description,
            "user_id": user_id,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            **kwargs,
        }
        self._tickets[ticket_id] = ticket
        return ticket

    def get(self, ticket_id: str) -> Optional[dict]:
        return self._tickets.get(ticket_id)

    def list_by_user(self, user_id: str) -> list[dict]:
        return [t for t in self._tickets.values() if t["user_id"] == user_id]


# ============================================================
# 第八部分：完整的智能客服 Agent
# ============================================================

class CustomerServiceAgent:
    """
    智能客服 Agent —— 整合所有模块的核心类。

    完整处理流水线：
    用户消息 → 情绪检测 → 意图分类 → 智能路由 → 执行动作 → 返回响应

    支持的配置：
    - use_llm: 是否使用 LLM（true 需要 API Key）
    - use_rag: 是否使用向量数据库 RAG
    """

    def __init__(self, use_llm: bool = False, use_rag: bool = False):
        """
        初始化 Agent。

        Args:
            use_llm: 是否使用 LLM（情绪分析、回答生成）
            use_rag: 是否使用向量数据库 RAG
        """
        # 初始化各模块
        self.emotion_detector = EmotionDetector(use_llm=use_llm)
        self.intent_classifier = IntentClassifier()
        self.router = SmartRouter()
        self.rag_engine = RAGEngine(use_vector_db=use_rag)
        self.memory = ConversationMemory(window_size=10)
        self.ticket_manager = SimpleTicketManager()

        # LLM（延迟初始化）
        self._llm = None
        self.use_llm = use_llm

        # 会话记录（用于复盘）
        self._session_logs: dict[str, list] = {}

        print(f"  [Agent] 初始化完成 (LLM={use_llm}, RAG={use_rag})")

    @property
    def llm(self):
        """延迟加载 LLM 实例。"""
        if self._llm is None and self.use_llm:
            try:
                from config import get_llm
                self._llm = get_llm(temperature=0.5)
            except Exception as e:
                print(f"  [警告] LLM 初始化失败: {e}")
                self.use_llm = False
        return self._llm

    def chat(self, session_id: str, user_message: str, user_id: str = "anonymous") -> AgentResponse:
        """
        处理用户消息并返回响应。

        这是 Agent 的主入口方法，完整执行：
        情绪检测 → 意图分类 → 智能路由 → 执行动作 → 记录记忆

        Args:
            session_id: 会话 ID
            user_message: 用户消息
            user_id: 用户 ID

        Returns:
            AgentResponse: Agent 的响应
        """
        start_time = datetime.now()

        # 初始化会话日志
        if session_id not in self._session_logs:
            self._session_logs[session_id] = []

        # 记录用户消息
        self.memory.add_message(session_id, "user", user_message)

        # ===== Step 1: 情绪检测 =====
        emotion_result = self.emotion_detector.detect(user_message)
        emotion = emotion_result["emotion"]
        intensity = emotion_result["intensity"]

        # ===== Step 2: 意图分类 =====
        intent_result = self.intent_classifier.classify(user_message)
        primary_intent = intent_result["primary_intent"]
        confidence = intent_result["confidence"]

        # ===== Step 3: 智能路由 =====
        routing = self.router.route(emotion_result, intent_result)
        action = routing["action"]
        emotion_prefix = routing.get("emotion_prefix", "")

        # ===== Step 4: 执行动作 =====
        reply = ""
        ticket_id = None
        escalation_id = None

        if action == RouteAction.GREETING:
            reply = self._handle_greeting(user_message)

        elif action == RouteAction.RAG_QUERY:
            reply = self._handle_rag_query(user_message, session_id)

        elif action == RouteAction.CREATE_TICKET:
            reply, ticket_id = self._handle_create_ticket(
                user_message, user_id, primary_intent, emotion_result, routing["priority"]
            )

        elif action == RouteAction.ESCALATE:
            reply, escalation_id = self._handle_escalation(
                user_message, session_id, user_id, emotion_result, routing["priority"]
            )

        else:  # LLM_FALLBACK
            reply = self._handle_fallback(user_message, session_id)

        # 添加情绪安抚前缀
        if emotion_prefix and not reply.startswith(emotion_prefix[:5]):
            reply = emotion_prefix + reply

        # ===== Step 5: 记录并返回 =====
        # 保存 AI 回复到记忆
        self.memory.add_message(session_id, "agent", reply,
                                action=action.value, emotion=emotion.value)

        # 记录会话日志
        self._session_logs[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "agent_reply": reply,
            "emotion": emotion.value,
            "intent": primary_intent.value,
            "action": action.value,
            "ticket_id": ticket_id,
            "escalation_id": escalation_id,
        })

        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return AgentResponse(
            reply=reply,
            session_id=session_id,
            emotion=emotion,
            intent=primary_intent,
            action=action,
            ticket_id=ticket_id,
            escalation_id=escalation_id,
            confidence=confidence,
            processing_time_ms=round(processing_time, 1),
        )

    def _handle_greeting(self, message: str) -> str:
        """处理问候。"""
        greetings = [
            "您好！我是星辰科技客服小星，请问有什么可以帮您？",
            "您好呀~ 我是星辰科技的智能客服小星，很高兴为您服务！",
        ]
        return greetings[hash(message) % len(greetings)]

    def _handle_rag_query(self, message: str, session_id: str) -> str:
        """处理知识库查询。"""
        # 获取对话上下文，优化查询
        history_text = self.memory.get_history_text(session_id)

        # 如果有 LLM，使用 LLM 结合知识库生成回答
        if self.llm:
            try:
                rag_answer = self.rag_engine.query(message)
                response = self.llm.invoke(f"""你是星辰科技客服小星。根据参考资料简洁回答用户问题。

对话历史：
{history_text}

参考资料：
{rag_answer}

用户问题：{message}

要求：回答简洁，3句以内。""")
                return response.content
            except Exception as e:
                print(f"  [警告] LLM 调用失败: {e}，降级到直接回答")

        # 无 LLM 模式：直接返回知识库答案
        return self.rag_engine.query(message)

    def _handle_create_ticket(self, message, user_id, intent, emotion, priority) -> tuple:
        """处理工单创建。"""
        intent_to_category = {
            IntentType.REFUND: "退款", IntentType.RETURN: "退货",
            IntentType.EXCHANGE: "换货", IntentType.REPAIR: "维修",
            IntentType.FEEDBACK: "反馈",
        }
        category = intent_to_category.get(intent, "其他")

        ticket = self.ticket_manager.create(
            category=category,
            title=f"{category}申请 - {message[:30]}",
            description=message,
            user_id=user_id,
            priority=priority,
            emotion=emotion.get("emotion", "neutral").value if isinstance(emotion.get("emotion"), EmotionType) else str(emotion.get("emotion", "")),
        )

        reply = (
            f"我已为您创建了{category}工单，工单号为 {ticket['ticket_id']}。\n"
            f"我们的客服团队会在 1-2 个工作日内与您联系处理。"
        )
        return reply, ticket["ticket_id"]

    def _handle_escalation(self, message, session_id, user_id, emotion, priority) -> tuple:
        """处理转人工。"""
        escalation_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"

        if priority == "urgent":
            reply = "非常抱歉！我已为您加急转接人工客服，坐席会尽快为您处理，请稍候。"
        else:
            reply = "好的，我正在为您转接人工客服，请稍候..."

        return reply, escalation_id

    def _handle_fallback(self, message: str, session_id: str) -> str:
        """兜底处理。"""
        if self.llm:
            try:
                history_text = self.memory.get_history_text(session_id)
                response = self.llm.invoke(f"""你是星辰科技客服小星。回答用户的问题。
如果不知道，诚实告知并建议拨打售后热线 400-888-9999。

对话历史：
{history_text}

用户问题：{message}

回答要简洁，3句以内。""")
                return response.content
            except Exception:
                pass

        return "感谢您的咨询！关于这个问题，我需要进一步核实。建议您拨打售后热线 400-888-9999（工作日 9:00-18:00）获取详细信息。"

    def get_session_log(self, session_id: str) -> list:
        """获取会话日志。"""
        return self._session_logs.get(session_id, [])

    def get_session_summary(self, session_id: str) -> dict:
        """获取会话摘要。"""
        log = self._session_logs.get(session_id, [])
        if not log:
            return {"session_id": session_id, "message_count": 0}

        emotions = [m["emotion"] for m in log]
        actions = [m["action"] for m in log]

        return {
            "session_id": session_id,
            "message_count": len(log),
            "emotions": dict(Counter(emotions)),
            "actions": dict(Counter(actions)),
            "has_ticket": any(m.get("ticket_id") for m in log),
            "has_escalation": any(m.get("escalation_id") for m in log),
            "duration_seconds": len(log) * 30,  # 估算
        }


# ============================================================
# 演示模式
# ============================================================

def demo():
    """演示完整 Agent 功能。"""
    print("=" * 60)
    print("  第十一章：智能客服 Agent 完整演示")
    print("=" * 60)

    # 初始化 Agent（use_llm=False 使用关键词模式，无需 API）
    agent = CustomerServiceAgent(use_llm=False, use_rag=False)
    print()

    # 模拟对话
    print("─" * 50)
    print("  模拟对话开始")
    print("─" * 50)

    conversations = [
        ("session-001", "user-alice", "你好"),
        ("session-001", "user-alice", "StarPods Pro 支持降噪吗？"),
        ("session-001", "user-alice", "多少钱？"),
        ("session-001", "user-alice", "好的谢谢！"),

        ("session-002", "user-bob", "我的耳机有杂音！退货！"),
        ("session-002", "user-bob", "StarPods Pro，买了两周了"),

        ("session-003", "user-charlie", "你们什么破客服！我要投诉！"),
        ("session-003", "user-charlie", "退款说了三遍了！！"),

        ("session-004", "user-dave", "StarWatch X 怎么测血氧？"),
        ("session-004", "user-dave", "谢谢，很清楚"),

        ("session-005", "user-eve", "帮我转人工"),
    ]

    for session_id, user_id, message in conversations:
        print(f"\n👤 [{user_id}]: {message}")

        response = agent.chat(session_id, message, user_id)
        print(f"🤖 [小星]: {response.reply}")
        print(f"   📊 情绪={response.emotion.value} | 意图={response.intent.value} | "
              f"动作={response.action.value} | {response.processing_time_ms}ms")

        if response.ticket_id:
            print(f"   📋 工单: {response.ticket_id}")
        if response.escalation_id:
            print(f"   🚨 转人工: {response.escalation_id}")

    # 展示会话摘要
    print("\n\n" + "=" * 50)
    print("  会话摘要")
    print("=" * 50)

    for session_id in ["session-001", "session-002", "session-003"]:
        summary = agent.get_session_summary(session_id)
        print(f"\n  📋 {session_id}:")
        print(f"     消息数: {summary['message_count']}")
        print(f"     情绪分布: {summary['emotions']}")
        print(f"     动作分布: {summary['actions']}")
        print(f"     工单: {'有' if summary['has_ticket'] else '无'}")
        print(f"     转人工: {'是' if summary['has_escalation'] else '否'}")

    print("\n\n✓ 第十一章核心 Agent 演示完成！")


if __name__ == "__main__":
    demo()
