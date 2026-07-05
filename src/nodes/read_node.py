from typing import Any, Dict, Optional

from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    ReadInput,
    ReadOutput,
    ReadingStrategy,
    State,
)
from src.nodes.base_node import BaseNode


class ReadNode(BaseNode[ReadInput, ReadOutput]):
    """阅读节点：阅读文档内容并提取关键信息。"""

    name = "read_node"
    input_model = ReadInput
    output_model = ReadOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def process(self, input_data: ReadInput) -> ReadOutput:
        """TODO: implement core reading logic.

        1. 根据 reading_strategy 遍历 documents。
        2. 对每个文档调用阅读/提取服务。
        3. 汇总关键信息并返回。
        """
        self.logger.info(f"[{self.name}] documents_count={len(input_data.documents)}")

        # 占位输出
        placeholder = ReadOutput(
            key_info=[],
            status="completed",
            failed_paper_ids=[],
        )
        return placeholder


async def read_node(state: State) -> State:
    """LangGraph 适配函数：阅读节点。"""
    state_queue = state["state_queue"]
    current_state = state["value"]

    try:
        current_state.current_step = ExecutionState.READING
        await state_queue.put(
            BackToFrontData(step=ExecutionState.READING, state="initializing", data=None)
        )

        node = ReadNode(current_state.config.get("read_node", {}))

        read_input = ReadInput(
            documents=current_state.search_output.results,
            reading_strategy=ReadingStrategy(
                focus_areas=["method", "result", "limitation"],
                depth="medium",
            ),
        )

        read_output = await node.run(read_input)
        current_state.read_output = read_output

        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.READING,
                state="completed",
                data=f"阅读完成，共提取 {len(read_output.key_info)} 条关键信息",
            )
        )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Reading failed: {str(e)}"
        current_state.error.read_node_error = err_msg
        await state_queue.put(
            BackToFrontData(step=ExecutionState.READING, state="error", data=err_msg)
        )
        return {"value": current_state}
