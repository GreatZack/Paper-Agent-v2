"""原生 OpenAI 兼容客户端薄封装。

替换 AutoGen(autogen_ext/agentchat) 的 OpenAIChatCompletionClient，
保留原有代码真正用到的极小接口：

- create(messages, json_output) -> LLMResult(content=...)
- close()
- SystemMessage / UserMessage（轻量消息对象）

内部对 AsyncOpenAI 采用懒初始化，close() 仅释放连接，后续 create 可重新建立，
以便在免费实例（512MB/0.1CPU）上安全复用共享实例而不丢失连接池。
"""

from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 1


class SystemMessage:
    """轻量系统消息（兼容旧 autogen 接口）。"""

    type = "system"

    def __init__(self, content: str):
        self.content = content
        self.source = None


class UserMessage:
    """轻量用户消息（兼容旧 autogen 接口）。"""

    type = "user"

    def __init__(self, content: Union[str, List[Any]], source: Optional[str] = None):
        self.content = content
        self.source = source


class LLMResult:
    """LLM 调用结果（兼容旧 autogen 的 result.content 访问方式）。"""

    def __init__(self, content: str):
        self.content = content


def _message_to_openai(
    message: Union[SystemMessage, UserMessage, Dict[str, Any]],
) -> Dict[str, Any]:
    """把轻量消息转换为 OpenAI Chat Completions 格式。"""
    if isinstance(message, dict):
        return message

    role = getattr(message, "type", "user")
    content = message.content

    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list):
        return {
            "role": role,
            "content": [{"type": "text", "text": str(part)} for part in content],
        }
    return {"role": role, "content": str(content)}


class OpenAICompatClient:
    """极薄封装 AsyncOpenAI，暴露 create()/close() 两个方法。

    - json_output=True 时优先使用 response_format={"type":"json_object"}；
      兼容后端不支持时自动降级为普通文本请求（保留 json 兜底解析）。
    - 内部 client 懒初始化，close() 仅断开连接。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **_: Any,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[AsyncOpenAI] = None

    def _ensure_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._client

    async def create(
        self,
        messages: List[Union[SystemMessage, UserMessage]],
        json_output: bool = False,
        **kwargs: Any,
    ) -> LLMResult:
        chat_messages = [_message_to_openai(m) for m in messages]
        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": chat_messages,
            **kwargs,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}

        client = self._ensure_client()
        try:
            resp = await client.chat.completions.create(**payload)
        except Exception:
            if not json_output:
                raise
            payload.pop("response_format", None)
            resp = await client.chat.completions.create(**payload)

        content = (resp.choices[0].message.content if resp.choices else "") or ""
        return LLMResult(content)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None