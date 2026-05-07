"""
===============================================
第三章 示例 3：RAG 问答系统（Milvus 版）
===============================================
目标：将检索到的知识库内容与 LLM 结合，构建准确的问答系统。

知识点：
    1. RetrievalQA 链：检索 + 生成的完整流程
    2. 自定义检索提示词
    3. 检索参数调优
    4. 带来源引用的回复

运行方式：
    cd ch03-rag
    python 03_rag_qa.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import get_llm, get_embeddings, get_milvus_config


# ============================================================
# RAG 提示词模板
# ============================================================
RAG_PROMPT_TEMPLATE = """你是「星辰科技」的客服助手。请根据以下检索到的知识库内容回答用户问题。

## 回答规则
1. 只基于下方提供的「参考资料」回答，不要编造信息
2. 如果参考资料中没有相关信息，诚实回答"抱歉，我没有找到相关信息，建议您拨打售后热线 400-888-9999"
3. 回答要简洁明了，控制在 3 句话以内
4. 如果涉及步骤，用编号列出

## 参考资料
{context}

## 用户问题
{question}

## 回答
"""

# 带来源引用的提示词
RAG_PROMPT_WITH_SOURCES = """你是「星辰科技」的客服助手。请根据参考资料回答用户问题，并在回答末尾标注信息来源。

## 回答规则
1. 基于参考资料回答，不编造信息
2. 回答简洁，3句以内
3. 回答末尾用「参考：xxx」标注来源

## 参考资料
{context}

## 用户问题
{question}

## 回答（含来源标注）
"""


def format_docs(docs):
    """将检索到的文档格式化为文本。"""
    return "\n\n---\n\n".join(
        f"[片段 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def get_or_build_vectorstore(embeddings, collection_name="customer_service_kb"):
    """
    加载已有的 Milvus 向量库，不存在则从文档创建。

    Milvus 的好处：只要 uri 和 collection_name 对得上，数据就一直在，
    不需要像 ChromaDB 那样手动管 persist_directory。
    """
    milvus_config = get_milvus_config()
    connection_args = {"uri": milvus_config["uri"]}

    try:
        # 尝试加载已有集合
        vectorstore = Milvus(
            embedding_function=embeddings,
            connection_args=connection_args,
            collection_name=collection_name,
        )
        # 试一下能不能检索，不能的话说明集合是空的
        vectorstore.similarity_search("测试", k=1)
        print("从已有 Milvus 集合加载...")
        return vectorstore
    except Exception:
        # 集合不存在或为空，从头创建
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


def main():
    print("=" * 60)
    print("  第三章 示例 3：RAG 问答系统（Milvus）")
    print("=" * 60)

    # --------------------------------------------------
    # 步骤 1：准备向量数据库
    # --------------------------------------------------
    print("\n--- 步骤 1：加载向量数据库 ---\n")

    embeddings = get_embeddings()
    vectorstore = get_or_build_vectorstore(embeddings)
    print(f"向量数据库就绪")

    # --------------------------------------------------
    # 步骤 2：构建 RAG 链（方式一：LCEL 手动组装）
    # --------------------------------------------------
    print("\n--- 步骤 2：构建 RAG 链 ---\n")

    llm = get_llm(temperature=0.3)  # RAG 场景用低温度，保证准确性

    # 创建检索器（从向量库检索 top_k 个相关文档）
    retriever = vectorstore.as_retriever(
        search_type="similarity",       # 相似度检索
        search_kwargs={"k": 3},          # 返回 top 3
    )

    # 构建 RAG 提示词
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 用 LCEL 组装完整的 RAG 链：
    # question → retriever(检索) → format_docs(格式化) → prompt(填充模板) → llm(生成) → 解析
    rag_chain = (
        {
            "context": retriever | format_docs,  # 检索并格式化
            "question": RunnablePassthrough(),    # 直接传递用户问题
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------
    # 步骤 3：测试 RAG 问答
    # --------------------------------------------------
    print("--- 步骤 3：RAG 问答测试 ---\n")

    test_questions = [
        "StarPods Pro 的降噪深度是多少？",
        "我的耳机连接不稳定怎么办？",
        "StarWatch X 支持游泳吗？",
        "退货需要多长时间？",
        "StarBox 能控制哪些品牌的智能设备？",
        "你们有没有卖手机？",  # 知识库中没有的信息
    ]

    for question in test_questions:
        print(f"👤 用户: {question}")

        # 调用 RAG 链
        answer = rag_chain.invoke(question)
        print(f"🤖 小星: {answer}")

        # 同时展示检索到的参考文档
        docs = retriever.invoke(question)
        print(f"   📚 检索到 {len(docs)} 条参考资料")
        for i, doc in enumerate(docs, 1):
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"      [{i}] {preview}...")
        print()

    # --------------------------------------------------
    # 步骤 4：带来源引用的回复
    # --------------------------------------------------
    print("--- 步骤 4：带来源引用的回复 ---\n")

    rag_prompt_sources = ChatPromptTemplate.from_template(RAG_PROMPT_WITH_SOURCES)
    rag_chain_sources = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_sources
        | llm
        | StrOutputParser()
    )

    question = "手表的心率监测准确吗？"
    answer = rag_chain_sources.invoke(question)
    print(f"👤 用户: {question}")
    print(f"🤖 小星: {answer}")

    print("""
    RAG 流程总结：

    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  用户问题   │────→│  检索器    │────→│  相关文档   │
    └───────────┘     └───────────┘     └─────┬─────┘
                                             │
    ┌───────────┐     ┌───────────┐          │
    │  最终答案   │←────│  LLM 生成  │←─────────┘
    └───────────┘     └───────────┘
          ↑                  ↑
          └──── 问题 + 文档 ──┘
    """)

    print("\n✓ 第三章示例 3 运行完成！")
    print("  下一课：04_full_rag_bot.py - 完整知识库客服")


if __name__ == "__main__":
    main()
