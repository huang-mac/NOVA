"""
===============================================
智能客服工单 Agent 系列 - 共享配置模块
===============================================
说明：本模块被所有章节共享引用，负责加载环境变量和初始化 LLM。
      每个章节的 Python 文件只需要 from config import llm, embeddings 即可。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件（从项目根目录）
env_path = os.path.join(os.path.dirname(__file__), "", ".env")
load_dotenv(env_path)


# ============================================================
# LLM 配置
# ============================================================
def get_llm_config():
    """
    获取 LLM 配置参数。

    优先从环境变量读取，如果未设置则使用默认值。
    支持 OpenAI 官方 API 或任何兼容 OpenAI 接口的服务（如 DeepSeek、智谱等）。

    Returns:
        dict: 包含 api_key, base_url, model_name 的配置字典
    """
    return {
        "api_key": os.getenv("OPENAI_API_KEY", "sk-your-key-here"),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


# ============================================================
# 初始化 LangChain LLM 实例（延迟加载）
# ============================================================
_llm_instance = None


def get_llm(temperature=0.7, max_tokens=2000):
    """
    获取 LangChain ChatOpenAI 实例（单例模式）。

    Args:
        temperature (float): 生成温度，值越大越随机。
                              客服场景建议 0.3-0.7，太低显得机械，太高容易跑题。
        max_tokens (int): 单次回复最大 token 数。

    Returns:
        ChatOpenAI: LangChain 的 LLM 封装实例
    """
    global _llm_instance

    # 如果全局实例已存在且参数匹配，直接复用
    if _llm_instance is not None:
        return _llm_instance

    from langchain_openai import ChatOpenAI

    config = get_llm_config()
    _llm_instance = ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model_name"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _llm_instance


# ============================================================
# Milvus 向量数据库配置
# ============================================================
def get_milvus_config():
    """
    获取 Milvus 向量数据库配置。

    本地开发用 Milvus Lite（文件模式，零部署）；
    生产环境改为连接 Milvus Server 或 Zilliz Cloud。

    Returns:
        dict: Milvus 连接配置
    """
    return {
        # "uri": os.getenv("MILVUS_URI", "./milvus_lite.db"),  # Milvus Lite 文件模式
        # 生产环境示例：
        "uri": os.getenv("MILVUS_URI", "http://114.132.151.31:19530"),
        # "token": os.getenv("MILVUS_TOKEN", ""),
    }


# ============================================================
# Embeddings 配置（独立于 LLM，支持混元等兼容服务）
# ============================================================
def get_embeddings_config():
    """
    获取 Embeddings 模型配置参数。

    支持任何兼容 OpenAI /embeddings 接口的服务，如：
      - 腾讯混元（hunyuan-embedding, 1024 维）
      - OpenAI（text-embedding-3-small, 1536 维）

    Returns:
        dict: 包含 api_key, base_url, model, dimension 的配置字典
    """
    return {
        "api_key": os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "sk-your-key-here")),
        "base_url": os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "dimension": int(os.getenv("EMBEDDING_DIMENSION", "1536")),
    }


# ============================================================
# 初始化 Embeddings 实例（用于后续 RAG 章节）
# ============================================================
_embeddings_instance = None


def get_embeddings():
    """
    获取 LangChain OpenAIEmbeddings 实例（单例模式）。

    Embeddings 用于将文本转换为向量，是 RAG 检索的核心组件。
    优先使用 EMBEDDING_* 环境变量，未设置则回退到 LLM 配置。

    Returns:
        OpenAIEmbeddings: 向量嵌入模型实例
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    from langchain_openai import OpenAIEmbeddings

    config = get_embeddings_config()
    _embeddings_instance = OpenAIEmbeddings(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        dimensions=config["dimension"],
        check_embedding_ctx_length=False,  # 混元等兼容服务不支持 token ID，需传原始文本
    )
    return _embeddings_instance


# ============================================================
# 快捷导入别名（方便各章节使用）
# ============================================================
# 使用方式：
#   from config import llm          # 获取默认 LLM 实例
#   from config import embeddings   # 获取默认 Embeddings 实例
#
# 注意：llm 是属性访问，每次调用 get_llm()

class _LazyLLM:
    """延迟加载的 LLM 代理，首次访问时才真正创建实例。"""
    def __call__(self, temperature=0.7, max_tokens=2000):
        return get_llm(temperature=temperature, max_tokens=max_tokens)

llm = _LazyLLM()
embeddings = get_embeddings

# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=== 配置测试 ===")
    config = get_llm_config()
    print(f"API Base URL: {config['base_url']}")
    print(f"Model: {config['model_name']}")
    print(f"API Key: {config['api_key'][:10]}...")

    # 测试 LLM 连接
    try:
        model = get_llm()
        response = model.invoke("你好，请用一句话介绍你自己。")
        print(f"\nLLM 响应测试: {response.content}")
        print("✓ 配置正确，LLM 可正常使用！")
    except Exception as e:
        print(f"✗ LLM 连接失败: {e}")
        print("  请检查 .env 文件中的 API Key 和 Base URL 是否正确。")
