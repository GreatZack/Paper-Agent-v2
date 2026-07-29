import asyncio
import threading
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.core.config import config
from src.core.state_models import (
    BackToFrontData,
    ConfigSchema,
    ExecutionState,
    NodeError,
    PaperAgentState,
    State,
)
from src.nodes.parse_node import parse_node
from src.nodes.read_node import read_node
from src.nodes.search_node import search_node
from src.nodes.write_node import write_node
from src.utils.log_utils import setup_logger


class WorkflowOrchestrator:
    """搜索-阅读-解析-撰写工作流编排器。"""

    def __init__(
        self,
        state_queue: Optional[asyncio.Queue] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.state_queue = state_queue or asyncio.Queue()
        self.config = config or {}
        self.logger = setup_logger("orchestrator", "orchestrator.log")

        # 流程控制标志
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()

        # 编译图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建并编译 LangGraph 工作流。"""
        builder = StateGraph(State, context_schema=ConfigSchema)

        # 添加节点
        builder.add_node("search_node", search_node)
        builder.add_node("read_node", read_node)
        builder.add_node("parse_node", parse_node)
        builder.add_node("write_node", write_node)
        builder.add_node("error_node", self.error_node)

        # 入口点
        builder.set_entry_point("search_node")
        builder.add_edge(START, "search_node")

        # 条件边
        builder.add_conditional_edges("search_node", self._route)
        builder.add_conditional_edges("read_node", self._route)
        builder.add_conditional_edges("parse_node", self._route)
        builder.add_conditional_edges("write_node", self._route)

        # 错误节点连接到 END
        builder.add_edge("error_node", END)

        return builder.compile()

    def _route(self, state: State) -> str:
        """条件路由函数。

        根据当前步骤、错误状态和流程控制标志决定下一跳目标。
        """
        current_state = state["value"]

        # 流程控制：停止或暂停均直接结束
        if self._stop_event.is_set() or self._pause_event.is_set():
            self.logger.info("Workflow control flag set, routing to END")
            return END

        err = current_state.error
        step = current_state.current_step

        if step == ExecutionState.SEARCHING and err.search_node_error is None:
            return "read_node"

        elif step == ExecutionState.READING:
            if err.read_node_error:
                return "error_node"

            # 检查是否启用验证
            verify_cfg = current_state.config.get("read_node", {}).get("verify", {})
            verify_enabled = verify_cfg.get("enabled", True)
            if not verify_enabled:
                return "parse_node"

            verify_results = current_state.verify_results or {}
            if not verify_results:
                # 首次进入或所有结果被清空
                # 如果 read_node 已执行过且全部论文读取失败，则终止流程
                read_output = current_state.read_output
                if read_output.failed_paper_ids and len(read_output.key_info) == 0:
                    total = len(current_state.search_output.results)
                    failed = len(read_output.failed_paper_ids)
                    if total > 0 and failed >= total:
                        err.read_node_error = (
                            f"所有论文读取失败 ({failed}/{total})，"
                            f"可能是 PDF 未下载或路径无效"
                        )
                        return "error_node"
                return "read_node"

            max_retries = int(verify_cfg.get("max_retries_per_paper", 3))
            any_failed = False

            for paper_id, result in verify_results.items():
                if not result.passed:
                    any_failed = True
                    if result.retry_count >= max_retries:
                        failed_fields = [
                            item.field for item in result.items if not item.verified
                        ]
                        err.read_node_error = (
                            f"论文 {paper_id} 验证失败，已达到最大重试次数 "
                            f"({result.retry_count}/{max_retries})；"
                            f"未通过字段：{', '.join(failed_fields) or '未知'}"
                        )
                        self.logger.error(
                            f"[_route] 论文 {paper_id} 验证超限 "
                            f"(retry_count={result.retry_count}/{max_retries})，路由到 error_node"
                        )
                        return "error_node"

            if any_failed:
                # 有论文未通过但未超限，回 read_node 重试
                self.logger.info(
                    "[_route] 部分论文验证未通过，回 read_node 重试"
                )
                return "read_node"

            # 全部通过
            return "parse_node"

        elif step == ExecutionState.PARSING and err.parse_node_error is None:
            return "write_node"
        elif step == ExecutionState.WRITING and err.write_node_error is None:
            return END
        else:
            return "error_node"

    async def error_node(self, state: State) -> State:
        """错误处理节点。"""
        state_queue = state["state_queue"]
        current_state = state["value"]
        current_state.current_step = ExecutionState.FAILED

        err_msg = f"Workflow failed at {current_state.current_step}: {current_state.error.model_dump()}"
        self.logger.error(err_msg)

        await state_queue.put(
            BackToFrontData(step=ExecutionState.FAILED, state="error", data=err_msg)
        )
        return {"value": current_state}

    async def initialize(
        self,
        user_request: str,
        max_papers: int = None,
        **kwargs: Any,
    ) -> PaperAgentState:
        if max_papers is None:
            max_papers = int(config.get("default_max_papers", 50))
        """初始化工作流状态。"""
        initial_state = PaperAgentState(
            user_request=user_request,
            max_papers=max_papers,
            error=NodeError(),
            config={**self.config, **kwargs},
        )
        self.logger.info(f"Initialized workflow state for request: {user_request}")
        return initial_state

    async def start(
        self,
        user_request: str,
        max_papers: int = None,
        **kwargs: Any,
    ) -> PaperAgentState:
        """启动并执行完整工作流。"""
        self._reset_control_flags()
        initial_state = await self.initialize(user_request, max_papers, **kwargs)

        self.logger.info("Starting workflow...")
        await self.state_queue.put(
            BackToFrontData(step=ExecutionState.INITIALIZING, state="started", data=None)
        )

        final_state = await self.graph.ainvoke(
            {"state_queue": self.state_queue, "value": initial_state}
        )

        current_state = final_state["value"]
        if current_state.current_step not in (ExecutionState.FAILED, ExecutionState.STOPPED):
            current_state.current_step = ExecutionState.COMPLETED
            await self.state_queue.put(
                BackToFrontData(step=ExecutionState.COMPLETED, state="finished", data=None)
            )
            self.logger.info("Workflow completed successfully")
        else:
            self.logger.info(f"Workflow ended with state: {current_state.current_step}")

        return current_state

    def pause(self) -> None:
        """暂停工作流（下一次路由时结束）。"""
        self._pause_event.set()
        self.logger.info("Pause requested")

    def stop(self) -> None:
        """终止工作流（下一次路由时结束）。"""
        self._stop_event.set()
        self.logger.info("Stop requested")

    def resume(self) -> None:
        """清除暂停标志。"""
        self._pause_event.clear()
        self.logger.info("Resume requested")

    def _reset_control_flags(self) -> None:
        """重置流程控制标志。"""
        self._pause_event.clear()
        self._stop_event.clear()

    @staticmethod
    def get_status(state: State) -> ExecutionState:
        """获取当前工作流状态。"""
        return state["value"].current_step
