from typing import Any, Dict, Optional

from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    ParseInput,
    ParseOutput,
    ParseRule,
    State,
)
from src.nodes.base_node import BaseNode


class ParseNode(BaseNode[ParseInput, ParseOutput]):
    """解析节点：将关键信息按规则解析为结构化数据。"""

    name = "parse_node"
    input_model = ParseInput
    output_model = ParseOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    async def process(self, input_data: ParseInput) -> ParseOutput:
        """TODO: implement core parsing logic.

        1. 根据 parse_rules 对 content 进行字段抽取与类型转换。
        2. 构建结构化字典。
        3. 返回结构化数据与解析状态。
        """
        self.logger.info(f"[{self.name}] rules_count={len(input_data.parse_rules)}")

        # 占位输出
        placeholder = ParseOutput(
            structured_data={"placeholder": True},
            status="completed",
            unmatched_items=[],
        )
        return placeholder


async def parse_node(state: State) -> State:
    """LangGraph 适配函数：解析节点。"""
    state_queue = state["state_queue"]
    current_state = state["value"]

    try:
        current_state.current_step = ExecutionState.PARSING
        await state_queue.put(
            BackToFrontData(step=ExecutionState.PARSING, state="initializing", data=None)
        )

        node = ParseNode(current_state.config.get("parse_node", {}))

        parse_input = ParseInput(
            content=current_state.read_output.key_info,
            parse_rules=[
                ParseRule(name="core_problem", field_path="core_problem", data_type="string", required=True),
                ParseRule(name="key_methodology", field_path="key_methodology", data_type="string", required=True),
                ParseRule(name="main_results", field_path="main_results", data_type="string", required=False),
            ],
        )

        parse_output = await node.run(parse_input)
        current_state.parse_output = parse_output

        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.PARSING,
                state="completed",
                data=f"解析完成，结构化字段数: {len(parse_output.structured_data)}",
            )
        )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Parsing failed: {str(e)}"
        current_state.error.parse_node_error = err_msg
        await state_queue.put(
            BackToFrontData(step=ExecutionState.PARSING, state="error", data=err_msg)
        )
        return {"value": current_state}
