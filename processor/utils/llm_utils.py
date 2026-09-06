import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_llm_client_cache = {}

def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例
    - model: 允许不同节点使用不同模型
    - json_mode: True 时要求输出 JSON
    """
    m = model or os.getenv('LLM_DEFAULT_MODEL')
    key = (m,json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    extra_body = {"enable_thinking": False}

    model_kwargs: dict = {}
    if json_mode: # 是不是json格式
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model = m,
        temperature=os.getenv('LLM_DEFAULT_TEMPERATURE'),
        api_key = os.getenv('OPENAI_API_KEY'),
        base_url = os.getenv('OPENAI_API_BASE'),
        extra_body=extra_body,
        model_kwargs=model_kwargs,
    )

    _llm_client_cache[key] = client
    return client