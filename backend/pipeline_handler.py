"""Pipeline handler: wraps WorkflowOrchestrator and streams state via WebSocket."""
import asyncio
import json
import logging
from typing import Any, Dict

from src.core.config import config as app_config
from src.graph.orchestrator import WorkflowOrchestrator
from src.core.state_models import BackToFrontData, ExecutionState

logger = logging.getLogger("pipeline_handler")


async def run_pipeline(query: str, max_papers: int, websocket) -> None:
    """Run the paper-agent pipeline and push state updates to the WebSocket.

    Args:
        query: User's research query.
        max_papers: Maximum number of papers to process.
        websocket: FastAPI WebSocket connection for streaming state.
    """
    queue: asyncio.Queue[BackToFrontData] = asyncio.Queue()

    # 从 config.yaml 加载配置传递给编排器（使 download_pdf=true 等生效）
    orchestrator_config: Dict[str, Any] = {}
    search_config = app_config.get("search_node")
    if search_config:
        orchestrator_config["search_node"] = {
            **search_config,
            "frontend_enabled": True,
        }
    else:
        orchestrator_config["search_node"] = {"frontend_enabled": True}
    read_config = app_config.get("read_node")
    if read_config:
        orchestrator_config["read_node"] = read_config
    orchestrator_config["default_max_papers"] = app_config.get("default_max_papers", 20)

    orchestrator = WorkflowOrchestrator(
        state_queue=queue,
        config=orchestrator_config,
    )

    task = asyncio.create_task(orchestrator.start(query, max_papers))

    completed = False
    while not completed:
        try:
            msg: BackToFrontData = await asyncio.wait_for(queue.get(), timeout=1.0)

            payload = {
                "step": msg.step if isinstance(msg.step, str) else msg.step.value,
                "state": msg.state if isinstance(msg.state, str) else str(msg.state),
                "data": msg.data,
            }

            # Send progress update
            await websocket.send_json(payload)

            if msg.step in (
                ExecutionState.COMPLETED,
                ExecutionState.FAILED,
                ExecutionState.STOPPED,
            ):
                completed = True

        except asyncio.TimeoutError:
            if task.done():
                completed = True

    # Get the final state and send the report
    final_state = task.result()

    if final_state.current_step == ExecutionState.COMPLETED:
        report = final_state.write_output.generated_text
        if report:
            await websocket.send_json(
                {
                    "step": "report",
                    "state": "completed",
                    "data": report,
                }
            )
        else:
            await websocket.send_json(
                {
                    "step": "report",
                    "state": "empty",
                    "data": "Pipeline completed but no report was generated.",
                }
            )
    elif final_state.current_step == ExecutionState.FAILED:
        err = final_state.error
        error_detail = err.model_dump() if err else "Unknown error"
        await websocket.send_json(
            {
                "step": "failed",
                "state": "error",
                "data": str(error_detail),
            }
        )
    else:
        await websocket.send_json(
            {
                "step": "stopped",
                "state": "finished",
                "data": f"Pipeline ended with state: {final_state.current_step}",
            }
        )
