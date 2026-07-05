from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from src.utils.log_utils import setup_logger

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseNode(ABC, Generic[InputT, OutputT]):
    """工作流节点抽象基类。

    子类需指定 name、input_model、output_model，并实现 process 方法。
    """

    name: str = "base_node"
    input_model: Type[InputT]
    output_model: Type[OutputT]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = setup_logger(self.name, f"{self.name}.log")

    @abstractmethod
    async def process(self, input_data: InputT) -> OutputT:
        """节点核心业务逻辑，子类需实现。

        Args:
            input_data: 符合 input_model 的输入数据。

        Returns:
            符合 output_model 的输出数据。
        """
        raise NotImplementedError("Subclasses must implement the process method.")

    async def run(self, input_data: InputT) -> OutputT:
        """运行节点，自动调用 process 并记录日志。"""
        self.logger.info(f"[{self.name}] start processing")
        output = await self.process(input_data)
        self.logger.info(f"[{self.name}] finish processing")
        return output
