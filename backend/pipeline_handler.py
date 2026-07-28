"""Pipeline handler: wraps WorkflowOrchestrator and streams state via WebSocket."""

import asyncio
import logging
from typing import Any

from src.core.config import config as app_config
from src.core.state_models import BackToFrontData, ExecutionState
from src.graph.orchestrator import WorkflowOrchestrator

logger = logging.getLogger("pipeline_handler")
HEARTBEAT_INTERVAL_SECONDS = 20.0


async def _send_json(websocket, payload: dict[str, Any]) -> bool:
    """Send a WebSocket payload and report whether the client is still connected."""
    try:
        await websocket.send_json(payload)
        return True
    except (RuntimeError, OSError) as exc:
        logger.info("WebSocket disconnected while sending update: %s", exc)
        return False


async def run_pipeline(query: str, max_papers: int, websocket) -> None:
    """Run the paper-agent pipeline and push state updates to the WebSocket.

    Args:
        query: User's research query.
        max_papers: Maximum number of papers to process.
        websocket: FastAPI WebSocket connection for streaming state.
    """
    queue: asyncio.Queue[BackToFrontData] = asyncio.Queue()

    # 从 config.yaml 加载配置传递给编排器（使 download_pdf=true 等生效）
    orchestrator_config: dict[str, Any] = {}
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
    orchestrator_config["default_max_papers"] = app_config.get("default_max_papers", 5)

    orchestrator = WorkflowOrchestrator(
        state_queue=queue,
        config=orchestrator_config,
    )

    task = asyncio.create_task(orchestrator.start(query, max_papers))
    terminal_message_sent = False
    client_connected = True
    loop = asyncio.get_running_loop()
    next_heartbeat = loop.time() + HEARTBEAT_INTERVAL_SECONDS

    try:
        while True:
            try:
                wait_timeout = min(
                    1.0,
                    max(0.01, next_heartbeat - loop.time()),
                )
                msg: BackToFrontData = await asyncio.wait_for(
                    queue.get(), timeout=wait_timeout
                )

                payload = {
                    "step": msg.step if isinstance(msg.step, str) else msg.step.value,
                    "state": msg.state
                    if isinstance(msg.state, str)
                    else str(msg.state),
                    "data": msg.data,
                }

                # Send progress update
                client_connected = await _send_json(websocket, payload)
                if not client_connected:
                    break
                next_heartbeat = loop.time() + HEARTBEAT_INTERVAL_SECONDS

                if msg.step in (
                    ExecutionState.COMPLETED,
                    ExecutionState.FAILED,
                    ExecutionState.STOPPED,
                ):
                    terminal_message_sent = True
                    break

            except asyncio.TimeoutError:
                if task.done():
                    break
                if loop.time() < next_heartbeat:
                    continue
                client_connected = await _send_json(
                    websocket,
                    {
                        "step": "heartbeat",
                        "state": "running",
                        "data": "任务仍在处理中",
                    },
                )
                if not client_connected:
                    break
                next_heartbeat = loop.time() + HEARTBEAT_INTERVAL_SECONDS

        if not client_connected:
            return

        # A terminal queue message can arrive just before the task returns.
        final_state = await task

        if final_state.current_step == ExecutionState.COMPLETED:
            report = final_state.write_output.generated_text
            if report:
                await _send_json(
                    websocket,
                    {
                        "step": "report",
                        "state": "completed",
                        "data": report,
                    },
                )
            else:
                await _send_json(
                    websocket,
                    {
                        "step": "report",
                        "state": "empty",
                        "data": "Pipeline completed but no report was generated.",
                    },
                )
        elif final_state.current_step == ExecutionState.FAILED:
            if not terminal_message_sent:
                err = final_state.error
                error_detail = err.model_dump() if err else "Unknown error"
                await _send_json(
                    websocket,
                    {
                        "step": "failed",
                        "state": "error",
                        "data": str(error_detail),
                    },
                )
        elif not terminal_message_sent:
            await _send_json(
                websocket,
                {
                    "step": "stopped",
                    "state": "finished",
                    "data": f"Pipeline ended with state: {final_state.current_step}",
                },
            )
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
