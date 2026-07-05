from typing import Any, Dict, Optional

from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    State,
    WriteInput,
    WriteOutput,
)
from src.nodes.base_node import BaseNode


class WriteNode(BaseNode[WriteInput, WriteOutput]):
    """撰写节点：基于结构化数据生成最终文本。"""

    name = "write_node"
    input_model = WriteInput
    output_model = WriteOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def process(self, input_data: WriteInput) -> WriteOutput:
        """TODO: implement core writing logic.

        1. 根据 template 与 style_hints 组织 structured_data。
        2. 调用文本生成服务（LLM、模板引擎等）。
        3. 返回生成的文本与撰写状态。
        """
        self.logger.info(f"[{self.name}] structured_data_keys={list(input_data.structured_data.keys())}")

        # 占位输出
        placeholder = WriteOutput(
            generated_text="This is a placeholder generated text.",
            status="completed",
            section_map={"placeholder": "generated"},
        )
        return placeholder


async def write_node(state: State) -> State:
    """LangGraph 适配函数：撰写节点。"""
    state_queue = state["state_queue"]
    current_state = state["value"]

    try:
        current_state.current_step = ExecutionState.WRITING
        await state_queue.put(
            BackToFrontData(step=ExecutionState.WRITING, state="initializing", data=None)
        )

        node = WriteNode(current_state.config.get("write_node", {}))

        write_input = WriteInput(
            structured_data=current_state.parse_output.structured_data,
            template=current_state.config.get("write_template", "# Report\n\n{structured_data}"),
            style_hints={"language": "zh", "tone": "academic"},
        )

        write_output = await node.run(write_input)
        current_state.write_output = write_output

        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.WRITING,
                state="completed",
                data="撰写完成",
            )
        )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Writing failed: {str(e)}"
        current_state.error.write_node_error = err_msg
        await state_queue.put(
            BackToFrontData(step=ExecutionState.WRITING, state="error", data=err_msg)
        )
        return {"value": current_state}
