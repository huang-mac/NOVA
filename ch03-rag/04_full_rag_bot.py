"""
===============================================
第三章 示例 4：完整知识库客服（Milvus 版）
===============================================
目标：将 RAG 与多轮记忆结合，构建能检索知识库、记住对话历史的完整客服。

知识点：
    1. RAG + Memory 的结合方式
    2. 在检索 query 中融入对话上下文（Standalone Question Generation）
    3. 知识库未命中时的兜底策略

注意：LangChain 1.x 已移除 ConversationBufferWindowMemory，
      改为用列表手动管理滑动窗口记忆。

运行方式：
    cd ch03-rag
    python 04_full_rag_bot.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm, get_embeddings, get_milvus_config


# ============================================================
# 独立问题生成提示词
# ============================================================
# 核心问题：用户可能说"它支持吗？"、"那续航呢？"这样的追问
# 如果直接用追问去检索，向量数据库无法理解"它"指什么
# 所以需要先根据对话历史，将追问转换为独立完整的问题
STANDALONE_QUESTION_PROMPT = """根据对话历史和最新问题，生成一个独立的、完整的问题。
生成的问题应该包含对话中提到的具体产品名称和上下文信息。

对话历史：
{chat_history}

最新问题：{question}

独立问题："""

# RAG 回答提示词
RAG_ANSWER_PROMPT = """你是「星辰科技」的客服小星。根据参考资料回答用户问题。

## 规则
1. 基于参考资料回答，不编造
2. 如果资料中没有答案，说"抱歉，我暂时没有这方面的信息，建议您拨打售后热线 400-888-9999"
3. 简洁回答，3句以内
4. 涉及步骤时用编号列出
5. 结合对话历史，用自然连贯的方式回复

## 对话历史
{chat_history}

## 参考资料
{context}

## 用户问题
{question}

## 回答"""


def format_docs(docs):
    """格式化检索到的文档。"""
    if not docs:
        return "（未检索到相关参考资料）"
    return "\n\n".join(
        f"[{i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )


def format_chat_history(history_key="chat_history"):
    """将记忆中的对话历史格式化为文本。"""
    def _formatter(inputs):
        history = inputs.get(history_key, [])
        if not history:
            return "（无历史对话）"
        lines = []
        for msg in history:
            role = "用户" if msg.type == "human" else "客服小星"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    return RunnableLambda(_formatter)


def get_or_build_vectorstore(embeddings, collection_name="customer_service_kb"):
    """
    加载已有 Milvus 集合，不存在则从文档创建。
    """
    milvus_config = get_milvus_config()
    connection_args = {"uri": milvus_config["uri"]}

    try:
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
        )
        vectorstore.similarity_search("测试", k=1)
        print("从已有 Milvus 集合加载...")
        return vectorstore
    except Exception:
        print("创建新的 Milvus 集合...")
        knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
        doc_path = os.path.join(knowledge_dir, "product_knowledge.md")
        loader = TextLoader(doc_path, encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        vectorstore = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
            drop_old=True,
        )
        return vectorstore


class RAGCustomerServiceBot:
    """
    结合 RAG 和多轮记忆的完整客服机器人。

    处理流程：
    1. 用户输入 → 生成独立问题（融入上下文）
    2. 独立问题 → 检索知识库 → 获取相关文档
    3. 独立问题 + 相关文档 + 对话历史 → LLM 生成回复
    4. 回复 + 用户输入 → 保存到记忆
    """

    def __init__(self, window_size: int = 5):
        self.llm = get_llm(temperature=0.3)
        self.embeddings = get_embeddings()

        # 加载 Milvus 向量数据库
        self.vectorstore = get_or_build_vectorstore(self.embeddings)

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        # 对话记忆：用列表手动管理，等价于 ConversationBufferWindowMemory
        self.window_size = window_size
        self.messages = []

        # 构建处理链
        self._build_chains()

    def _build_chains(self):
        """构建 RAG 处理链。"""
        # 独立问题生成链
        self.question_chain = (
            ChatPromptTemplate.from_template(STANDALONE_QUESTION_PROMPT)
            | self.llm
            | StrOutputParser()
        )

        # RAG 回答链
        self.answer_chain = (
            ChatPromptTemplate.from_template(RAG_ANSWER_PROMPT)
            | self.llm
            | StrOutputParser()
        )

    def chat(self, user_input: str) -> str:
        """处理用户输入并返回回复。"""
        if user_input.strip().lower() in ("退出", "quit"):
            return "感谢使用星辰科技客服，再见！"

        # 获取对话历史（当前窗口内的消息）
        history = list(self.messages)

        # 格式化对话历史
        history_text = "\n".join(
            f"{'用户' if m.type == 'human' else '客服小星'}: {m.content}"
            for m in history
        ) if history else "（无历史对话）"

        # Step 1: 生成独立问题
        standalone_question = self.question_chain.invoke({
            "chat_history": history_text,
            "question": user_input,
        })

        # Step 2: 检索知识库
        docs = self.retriever.invoke(standalone_question)
        context = format_docs(docs)

        # Step 3: 生成回答
        answer = self.answer_chain.invoke({
            "chat_history": history_text,
            "context": context,
            "question": standalone_question,
        })

        # Step 4: 保存到记忆（滑动窗口裁切）
        self.messages.append(HumanMessage(content=user_input))
        self.messages.append(AIMessage(content=answer))
        max_size = self.window_size * 2
        if len(self.messages) > max_size:
            self.messages = self.messages[-max_size:]

        return answer, standalone_question, docs


def demo():
    """演示 RAG 客服的完整对话流程。"""
    print("=" * 60)
    print("  第三章 示例 4：完整知识库客服（Milvus）")
    print("=" * 60)

    bot = RAGCustomerServiceBot(window_size=5)

    print("\n--- 演示：多轮对话 + 上下文追问 ---\n")

    # 模拟一段包含追问的对话
    conversations = [
        "你们有什么耳机产品？",
        "StarPods Pro 的降噪效果怎么样？",
        "那续航呢？",               # 追问：需要上下文理解"它"指 StarPods Pro
        "连接不稳定怎么办？",         # 追问：隐含了"耳机连接不稳定"
        "你们卖手机吗？",             # 知识库中没有的信息
    ]

    for user_input in conversations:
        print(f"👤 用户: {user_input}")

        answer, standalone_q, docs = bot.chat(user_input)
        print(f"🤖 小星: {answer}")
        print(f"   🔍 检索问题: {standalone_q}")
        print(f"   📚 命中文档: {len(docs)} 条")
        if docs:
            print(f"   📄 最好匹配: {docs[0].page_content[:50]}...")
        print()

    # 导出对话记录
    print("--- 对话记录导出 ---")
    for msg in bot.messages:
        role = "用户" if msg.type == "human" else "小星"
        print(f"  {role}: {msg.content[:60]}...")

    print("\n✓ 第三章示例 4 运行完成！")
    print("  下一章：ch04-emotion-routing - 情绪识别与智能路由")


if __name__ == "__main__":
    demo()
