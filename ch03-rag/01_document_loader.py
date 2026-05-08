"""
===============================================
第三章 示例 1：文档加载与切分
===============================================
目标：学习如何加载文档、切分成适合向量化的文本块。

知识点：
    1. LangChain 文档加载器（TextLoader, DirectoryLoader 等）
    2. 文本切分策略（RecursiveCharacterTextSplitter）
    3. chunk_size 和 chunk_overlap 参数调优

运行方式：
    cd ch03-rag
    python 01_document_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ch01-basics"))

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    print("=" * 60)
    print("  第三章 示例 1：文档加载与切分")
    print("=" * 60)

    knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")

    # --------------------------------------------------
    # 步骤 1：加载单个文档
    # --------------------------------------------------
    print("\n--- 步骤 1：加载单个文档 ---\n")

    doc_path = os.path.join(knowledge_dir, "product_knowledge.md")
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    print(f"文档路径: {doc_path}")
    print(f"文档数量: {len(documents)}")
    print(f"文档元数据: {documents[0].metadata}")
    print(f"总字符数: {len(documents[0].page_content)}")
    print(f"前 200 字符预览:")
    print(f"  {documents[0].page_content[:200]}...")

    # --------------------------------------------------
    # 步骤 2：文本切分
    # --------------------------------------------------
    print("\n\n--- 步骤 2：文本切分 ---\n")

    # RecursiveCharacterTextSplitter 是最常用的切分器
    # 它会按以下优先级尝试在分隔符处切分：\n\n > \n > 空格 > 任意字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 每个 chunk 最大 500 字符
        chunk_overlap=50,      # chunk 之间重叠 50 字符（保持上下文连贯）
        length_function=len,   # 用字符数计算长度
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", "，", " ", ""],
        # 中文场景优化：优先在标题、段落、句子处切分
    )

    chunks = text_splitter.split_documents(documents)

    print(f"切分参数: chunk_size=500, chunk_overlap=50")
    print(f"切分结果: {len(documents)} 个文档 → {len(chunks)} 个文本块")
    print()

    # 展示前 5 个 chunk
    for i, chunk in enumerate(chunks[:5], 1):
        print(f"[Chunk {i}] ({len(chunk.page_content)} 字符)")
        print(f"  {chunk.page_content[:100]}...")
        print()

    # --------------------------------------------------
    # 步骤 3：不同切分参数对比
    # --------------------------------------------------
    print("--- 步骤 3：不同切分参数对比 ---\n")

    configs = [
        {"chunk_size": 200, "chunk_overlap": 20, "label": "小块（精细检索）"},
        {"chunk_size": 500, "chunk_overlap": 50, "label": "中等（平衡）"},
        {"chunk_size": 1000, "chunk_overlap": 100, "label": "大块（保留完整段落）"},
    ]

    for cfg in configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            length_function=len,
        )
        chunks = splitter.split_documents(documents)
        avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"  {cfg['label']:20s} | "
              f"chunk_size={cfg['chunk_size']:4d} | "
              f"→ {len(chunks):3d} 个块 | "
              f"平均 {avg_len:.0f} 字符/块")

    print("""
    切分参数选择建议：
    ┌──────────────────────────────────────────────────┐
    │ chunk_size 太小 → 丢失上下文，检索结果不完整      │
    │ chunk_size 太大 → 检索不精确，无关信息干扰        │
    │                                                  │
    │ 客服场景推荐：chunk_size=500, overlap=50         │
    │ FAQ 场景推荐：chunk_size=200, overlap=20         │
    │ 技术文档推荐：chunk_size=1000, overlap=100       │
    └──────────────────────────────────────────────────┘
    """)

    print("✓ 第三章示例 1 运行完成！")
    print("  下一课：02_vector_store.py - 向量存储与检索")


if __name__ == "__main__":
    main()
