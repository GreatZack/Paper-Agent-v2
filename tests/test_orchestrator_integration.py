import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.state_models import ExecutionState
from src.graph.orchestrator import WorkflowOrchestrator


@pytest.mark.asyncio
async def test_full_workflow_with_mocked_search():
    """集成测试：验证 search_node 与 LangGraph 编排器能协同完成完整工作流。"""
    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(
        state_queue=state_queue,
        config={"read_node": {"verify": {"enabled": False}}},
    )

    fake_paper = {
        "paper_id": "1234.56789",
        "title": "Integration Paper",
        "authors": ["Author A"],
        "summary": "Integration summary.",
        "published": 2024,
        "published_date": "2024-01-01T00:00:00",
        "url": "http://arxiv.org/abs/1234.56789",
        "pdf_url": "http://arxiv.org/pdf/1234.56789",
        "primary_category": "cs.AI",
        "categories": ["cs.AI"],
        "doi": None,
    }
    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(return_value=[fake_paper])

    with (
        patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher),
        patch(
            "src.nodes.search_node.SearchNode._get_search_agent",
            return_value=MagicMock(),
        ),
        patch(
            "src.nodes.search_node.SearchNode._generate_search_query",
            new=AsyncMock(return_value='all:"test"'),
        ),
        patch(
            "src.nodes.search_node.SearchNode._filter_relevant_papers",
            new=AsyncMock(side_effect=lambda papers, *args: (papers, [])),
        ),
        patch(
            "src.nodes.read_node.ReadNode.process", new_callable=AsyncMock
        ) as mock_read,
        patch(
            "src.nodes.parse_node.ParseNode.process", new_callable=AsyncMock
        ) as mock_parse,
        patch(
            "src.nodes.write_node.WriteNode.process", new_callable=AsyncMock
        ) as mock_write,
    ):
        mock_read.return_value = type(
            "ReadOutput",
            (),
            {"key_info": [], "status": "completed", "failed_paper_ids": []},
        )()
        mock_parse.return_value = type(
            "ParseOutput",
            (),
            {
                "papers": {},
                "raw_extractions": {},
                "taxonomy": {},
                "comparison_points": [],
                "discrepancies": [],
                "coverage": {},
                "status": "completed",
            },
        )()
        mock_write.return_value = type(
            "WriteOutput",
            (),
            {
                "generated_text": "Final report.",
                "status": "completed",
                "section_map": {},
            },
        )()

        final_state = await orchestrator.start(
            user_request="大语言模型在自动驾驶中的应用现状",
            max_papers=3,
        )

    assert final_state.current_step == ExecutionState.COMPLETED
    assert final_state.search_output.total_count == 1
    assert final_state.search_output.results[0].paper_id == "1234.56789"
    assert final_state.error.search_node_error is None


@pytest.mark.asyncio
async def test_workflow_routes_to_error_on_search_failure():
    """集成测试：验证搜索失败时工作流正确进入错误节点。"""
    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(state_queue=state_queue)

    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(side_effect=Exception("arxiv unavailable"))

    with (
        patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher),
        patch(
            "src.nodes.search_node.SearchNode._get_search_agent",
            return_value=MagicMock(),
        ),
        patch(
            "src.nodes.search_node.SearchNode._generate_search_query",
            new=AsyncMock(return_value='all:"test"'),
        ),
    ):
        final_state = await orchestrator.start(user_request="test", max_papers=3)

    assert final_state.current_step == ExecutionState.FAILED
    assert final_state.error.search_node_error is not None
    assert "arxiv unavailable" in final_state.error.search_node_error
