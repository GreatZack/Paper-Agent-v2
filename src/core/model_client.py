from typing import Any, Optional

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from src.core.config import config
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class ModelClient:
    """OpenAI 兼容远程模型客户端封装。"""

    @staticmethod
    def create_client(
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        vision: bool = True,
        function_calling: bool = True,
        json_output: bool = True,
        structured_output: bool = True,
        family: str = "Qwen",
    ) -> OpenAIChatCompletionClient:
        """根据提供商和模型名创建远程 API 模型客户端。

        Args:
            provider: 模型提供商名称，对应 config 中的配置节。
            model: 模型名称，例如 "Qwen/Qwen3-32B"。
            api_key: 可选，直接传入 API 密钥；未传入时从 config 读取。
            base_url: 可选，直接传入 API 基础地址；未传入时从 config 读取。
            vision: 是否支持视觉输入。
            function_calling: 是否支持函数调用。
            json_output: 是否支持 JSON 输出。
            structured_output: 是否支持结构化输出。
            family: 模型家族名称。

        Returns:
            配置完成的 OpenAIChatCompletionClient 实例。

        Raises:
            ValueError: 当 api_key 或 base_url 未配置时抛出。
        """
        provider_config = config.get(provider, {}) or {}
        api_key = api_key or provider_config.get("api_key")
        base_url = base_url or provider_config.get("base_url")

        if not api_key:
            raise ValueError(f"未配置 {provider} 的 api_key")
        if not base_url:
            raise ValueError(f"未配置 {provider} 的 base_url")

        model_info = ModelInfo(
            vision=vision,
            function_calling=function_calling,
            json_output=json_output,
            family=family,
            structured_output=structured_output,
        )

        return OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info=model_info,
        )


def create_model_client(client_type: str) -> OpenAIChatCompletionClient:
    """根据 client_type 从配置中创建远程模型客户端。

    Args:
        client_type: 配置节名称，例如 "search-model"、"reading-model"。

    Returns:
        配置完成的 OpenAIChatCompletionClient 实例。

    Raises:
        ValueError: 当配置缺失或参数无效时直接抛出，不做任何兜底处理。
    """
    model_config = config.get(client_type, {}) or {}
    provider = model_config.get("model-provider")
    model = model_config.get("model")

    if not provider or not model:
        model_config = config.get("default-model", {}) or {}
        provider = model_config.get("model-provider")
        model = model_config.get("model")

    if not provider or not model:
        raise ValueError(f"未配置 {client_type} 也未配置 default-model 的 model-provider 或 model")

    return ModelClient.create_client(
        provider=provider,
        model=model,
        api_key=model_config.get("api_key"),
        base_url=model_config.get("base_url"),
    )


def create_search_model_client() -> OpenAIChatCompletionClient:
    """创建用于搜索的远程模型客户端。"""
    return create_model_client("search-model")


# =============================================================================
# 本地模型调用预留接口
# =============================================================================
# 下面的代码段与本地模型调用相关，目前默认使用远程 API，因此相关实现暂不启用。
# 当需要接入 vLLM、Ollama、Transformers 等本地推理方案时，可在此扩展实现。

# def create_local_model_client(
#     model_path: str,
#     device: str = "auto",
#     load_in_8bit: bool = False,
#     **kwargs: Any,
# ) -> OpenAIChatCompletionClient:
#     """创建本地模型客户端。
#
#     Args:
#         model_path: 本地模型路径或 HuggingFace 模型 ID。
#         device: 运行设备，例如 "auto"、"cpu"、"cuda:0"。
#         load_in_8bit: 是否使用 8bit 量化加载。
#         **kwargs: 额外推理参数，例如 temperature、max_tokens。
#
#     Returns:
#         与远程模型接口一致的 OpenAIChatCompletionClient 实例。
#     """
#     raise NotImplementedError("本地模型调用功能尚未实现")


def create_local_model_client(
    model_path: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    **kwargs: Any,
) -> OpenAIChatCompletionClient:
    """本地模型客户端标准化接口（预留实现）。

    该接口目前仅作为预留，未来可接入 vLLM、Ollama、Transformers 等本地推理方案。
    参数和返回值类型已固定，后续实现需保持兼容。

    Args:
        model_path: 本地模型路径或 HuggingFace 模型 ID。
        device: 运行设备，例如 "auto"、"cpu"、"cuda:0"。
        load_in_8bit: 是否使用 8bit 量化加载。
        **kwargs: 额外推理参数，例如 temperature、max_tokens。

    Returns:
        与远程模型接口一致的 OpenAIChatCompletionClient 实例。

    Raises:
        NotImplementedError: 当前尚未实现本地模型调用，请使用远程 API 模型。
    """
    raise NotImplementedError("本地模型调用功能尚未实现，请使用远程 API 模型。")
