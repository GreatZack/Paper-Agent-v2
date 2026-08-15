import re
from functools import lru_cache
from typing import Any

from src.core.config import config
from src.core.openai_client import OpenAICompatClient
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

_ENV_PLACEHOLDER = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")


def _missing_setting(value: Any) -> bool:
    """Return whether a required model setting is empty or unresolved."""
    return (
        not isinstance(value, str)
        or not value.strip()
        or bool(_ENV_PLACEHOLDER.fullmatch(value.strip()))
    )


def _as_bool(value: Any, default: bool) -> bool:
    """Parse YAML booleans and string values consistently."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


class ModelClient:
    """OpenAI 兼容远程模型客户端封装（原生 openai SDK 薄壳）。"""

    @staticmethod
    def create_client(
        model: str,
        api_key: str,
        base_url: str,
        vision: bool = False,
        function_calling: bool = True,
        json_output: bool = True,
        structured_output: bool = True,
        family: str = "unknown",
    ) -> OpenAICompatClient:
        """创建统一的 OpenAI 兼容远程模型客户端。

        Args:
            model: 模型名称，例如 "Qwen/Qwen3-32B"。
            api_key: 模型服务 API 密钥。
            base_url: OpenAI 兼容 API 基础地址。
            vision: 是否支持视觉输入。
            function_calling: 是否支持函数调用。
            json_output: 是否支持 JSON 输出。
            structured_output: 是否支持结构化输出。
            family: 模型家族名称。

        Returns:
            配置完成的 OpenAICompatClient 实例。

        Raises:
            ValueError: 当 model、api_key 或 base_url 未配置时抛出。
        """
        if _missing_setting(model):
            raise ValueError("未配置 model.name")
        if _missing_setting(api_key):
            raise ValueError("未配置 model.api_key")
        if _missing_setting(base_url):
            raise ValueError("未配置 model.base_url")

        return OpenAICompatClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )


@lru_cache(maxsize=1)
def create_model_client() -> OpenAICompatClient:
    """从唯一的 ``model`` 配置创建所有节点共用的模型客户端。

    使用 lru_cache(1) 复用同一个无状态客户端实例，避免每请求新建
    多个 httpx 连接池（0.1 CPU/512MB 下是主要的内存浪费源）。
    """
    model_config = config.get("model", {}) or {}
    return ModelClient.create_client(
        model=model_config.get("name"),
        api_key=model_config.get("api_key"),
        base_url=model_config.get("base_url"),
        vision=_as_bool((model_config.get("capabilities") or {}).get("vision"), False),
        function_calling=_as_bool(
            (model_config.get("capabilities") or {}).get("function_calling"), True
        ),
        json_output=_as_bool(
            (model_config.get("capabilities") or {}).get("json_output"), True
        ),
        structured_output=_as_bool(
            (model_config.get("capabilities") or {}).get("structured_output"), True
        ),
        family=str((model_config.get("capabilities") or {}).get("family", "unknown")),
    )


def create_local_model_client(
    model_path: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    **kwargs: Any,
) -> OpenAICompatClient:
    """本地模型客户端标准化接口（预留实现）。

    该接口目前仅作为预留，未来可接入 vLLM、Ollama、Transformers 等本地推理方案。
    参数和返回值类型已固定，后续实现需保持兼容。

    Args:
        model_path: 本地模型路径或 HuggingFace 模型 ID。
        device: 运行设备，例如 "auto"、"cuda:0"。
        load_in_8bit: 是否使用 8bit 量化加载。
        **kwargs: 额外推理参数，例如 temperature、max_tokens。

    Returns:
        与远程模型接口一致的 OpenAICompatClient 实例。

    Raises:
        NotImplementedError: 当前尚未实现本地模型调用，请使用远程 API 模型。
    """
    raise NotImplementedError("本地模型调用功能尚未实现，请使用远程 API 模型。")
