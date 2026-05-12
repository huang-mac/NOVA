"""
===============================================
第三章 示例 2：向量存储与检索（Milvus 版）
===============================================
目标：将文档向量化存入 Milvus，并实现相似度检索。

知识点：
    1. Embeddings 将文本转换为高维向量
    2. Milvus 是生产级向量数据库，支持亿级数据、毫秒级检索
    3. Milvus Lite 模式：零部署，本地文件即可运行，开发调试很方便
    4. 生产环境可无缝切换到 Milvus Server 或 Zilliz Cloud
    5. 相似度检索：根据向量距离找到最相关的文档片段

运行方式：
    cd ch03-rag
    python 02_vector_store.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from config import get_embeddings, get_milvus_config


def main():
    print("=" * 60)
    print("  第三章 示例 2：向量存储与检索（Milvus）")
    print("=" * 60)

    # --------------------------------------------------
    # 步骤 1：加载并切分文档
    # --------------------------------------------------
    print("\n--- 步骤 1：加载知识库文档 ---\n")

    knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
    doc_path = os.path.join(knowledge_dir, "product_knowledge.md")

    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"知识库文档已加载并切分为 {len(chunks)} 个文本块")

    # --------------------------------------------------
    # 步骤 2：创建 Milvus 向量存储
    # --------------------------------------------------
    print("\n--- 步骤 2：向量化并存入 Milvus ---\n")

    # 获取 Embeddings 模型
    embeddings = get_embeddings()

    # 获取 Milvus 配置
    milvus_config = get_milvus_config()
    collection_name = "customer_service_kb"

    # 创建 Milvus 向量存储
    # langchain_milvus 会自动完成以下操作：
    #   1. 如果集合不存在，自动创建（字段：id, text, vector）
    #   2. 将每个 chunk 通过 embeddings 模型转换为向量
    #   3. 插入向量到 Milvus 集合
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"uri": milvus_config["uri"]},  # Milvus Lite 文件模式
        collection_name=collection_name,
        drop_old=True,  # 每次运行清空旧数据（生产环境改为 False）
    )

    print(f"向量数据库已创建: {milvus_config['uri']}")
    print(f"集合名称: {collection_name}")
    print(f"向量数量: {len(chunks)}")

    # --------------------------------------------------
    # 步骤 3：相似度检索
    # --------------------------------------------------
    print("\n--- 步骤 3：相似度检索测试 ---\n")

    test_queries = [
        "耳机连接不稳定怎么办",
        "手表的续航时间是多久",
        "音箱支持哪些智能家居品牌",
        "退货流程是什么",
        "怎么联系人工客服",
    ]

    for query in test_queries:
        print(f"🔍 查询: 「{query}」")
        print(f"   {'─' * 40}")

        # similarity_search：返回最相似的 k 个文档
        results = vectorstore.similarity_search(query, k=2)

        for i, doc in enumerate(results, 1):
            # 截断显示
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"   [{i}] {preview}...")

        print()

    # --------------------------------------------------
    # 步骤 4：带分数的检索
    # --------------------------------------------------
    print("--- 步骤 4：带相似度分数的检索 ---\n")

    query = "降噪效果不好怎么解决"
    print(f"🔍 查询: 「{query}」\n")

    # similarity_search_with_score 返回文档和相似度分数
    # Milvus 默认使用 IP（内积）相似度，分数越大越相似
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"   [{i}] 分数={score:.4f} | {preview}...")

    # --------------------------------------------------
    # 步骤 5：从已有向量存储加载（不需要重新计算）
    # --------------------------------------------------
    print("\n\n--- 步骤 5：从已有向量存储加载 ---\n")

    # 直接加载已存在的 Milvus 集合，不需要重新算向量
    # 只要 uri 和 collection_name 一致，数据就还在
    loaded_vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": milvus_config["uri"]},
        collection_name=collection_name,
    )

    # 直接使用
    results = loaded_vectorstore.similarity_search("StarPods Pro 的价格", k=2)
    print(f"🔍 从持久化数据库检索「StarPods Pro 的价格」:")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"   [{i}] {preview}...")

    print("""
    向量检索流程总结：

    用户问题 → Embeddings 转换为向量 → 在 Milvus 中做 ANN 近似最近邻搜索
                                                ↓
                                          返回最相关的文档片段

    Milvus vs ChromaDB：
    ┌──────────────────────────────────────────────────────┐
    │ Milvus Lite（本课程使用）  │ 文件模式，零部署，本地跑 │
    │ Milvus Server              │ 自建集群，支持亿级数据  │
    │ Zilliz Cloud               │ 托管服务，免运维       │
    │                                                       │
    │ 开发阶段用 Milvus Lite 就够了，上线时改一下 URI      │
    │ 就能无缝切换到 Milvus Server 或 Zilliz Cloud。       │
    └──────────────────────────────────────────────────────┘
    """)

    print("\n✓ 第三章示例 2 运行完成！")
    print("  下一课：03_rag_qa.py - RAG 问答系统")


if __name__ == "__main__":
    main()
