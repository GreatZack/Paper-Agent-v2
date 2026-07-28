import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline_handler import run_pipeline
from src.core.state_models import BackToFrontData, ExecutionState


@pytest.mark.asyncio
async def test_failed_terminal_message_waits_for_task_and_is_not_duplicated():
    class FakeOrchestrator:
        def __init__(self, state_queue, config):
            self.state_queue = state_queue

        async def start(self, query, max_papers):
            await self.state_queue.put(
                BackToFrontData(
                    step=ExecutionState.FAILED,
                    state="error",
                    data="search failed",
                )
            )
            await asyncio.sleep(0.01)
            return SimpleNamespace(
                current_step=ExecutionState.FAILED,
                error=SimpleNamespace(model_dump=lambda: {"search": "failed"}),
            )

    websocket = SimpleNamespace(send_json=AsyncMock())

    with patch("backend.pipeline_handler.WorkflowOrchestrator", FakeOrchestrator):
        await run_pipeline("test", 1, websocket)

    websocket.send_json.assert_awaited_once_with(
        {"step": "failed", "state": "error", "data": "search failed"}
    )


@pytest.mark.asyncio
async def test_completed_pipeline_sends_terminal_state_then_report():
    class FakeOrchestrator:
        def __init__(self, state_queue, config):
            self.state_queue = state_queue

        async def start(self, query, max_papers):
            await self.state_queue.put(
                BackToFrontData(
                    step=ExecutionState.COMPLETED,
                    state="finished",
                    data=None,
                )
            )
            return SimpleNamespace(
                current_step=ExecutionState.COMPLETED,
                write_output=SimpleNamespace(generated_text="final report"),
            )

    websocket = SimpleNamespace(send_json=AsyncMock())

    with patch("backend.pipeline_handler.WorkflowOrchestrator", FakeOrchestrator):
        await run_pipeline("test", 1, websocket)

    assert websocket.send_json.await_count == 2
    assert websocket.send_json.await_args_list[1].args[0] == {
        "step": "report",
        "state": "completed",
        "data": "final report",
    }
