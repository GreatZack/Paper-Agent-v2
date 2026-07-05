import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import arxiv
import pytest

from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    NodeError,
    PaperAgentState,
    SearchInput,
    SearchOutput,
    SearchResult,
    SearchScope,
    State,
)
from src.nodes.search_node import SearchNode, SearchQuery, parse_search_query, search_node


@pytest.fixture
def sample_papers():
    return [
        {
            "paper_id": "2411.11607v2",
            "title": "ROS2 Automated Driving",
            "authors": ["Alice", "Bob"],
            "summary": "Performance evaluation summary.",
            "published": 2024,
            "published_date": "2024-11-18T14:29:22+00:00",
            "url": "http://arxiv.org/abs/2411.11607v2",
            "pdf_url": "http://arxiv.org/pdf/2411.11607v2",
            "primary_category": "cs.RO",
            "categories": ["cs.RO"],
            "doi": None,
        }
    ]



@pytest.fixture
def mock_search_agent():
    agent = MagicMock()
    agent.run = AsyncMock()
    return agent


@pytest.mark.asyncio
async def test_search_node_process_without_llm(sample_papers):
    node = SearchNode(config={"use_llm": False})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)

    input_data = SearchInput(
        query_keywords=["ROS2", "automated driving"],
        search_scope=SearchScope(max_results=5),
        user_request="帮我搜索 ROS2 在自动驾驶中的应用",
    )
    output = await node.process(input_data)

    assert isinstance(output, SearchOutput)
    assert output.total_count == 1
    assert output.status == "completed"
    assert output.results[0].paper_id == "2411.11607v2"
    assert output.results[0].pdf_url == "http://arxiv.org/pdf/2411.11607v2"
    node.paper_searcher.search_papers.assert_awaited_once_with(
        querys=["ROS2", "automated driving"],
        max_results=5,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        start_date=None,
        end_date=None,
    )


@pytest.mark.asyncio
async def test_search_node_process_with_llm_generated_queries(sample_papers):
    generated = SearchQuery(querys=["LLM", "autonomous driving"], start_date="2023-01-01", end_date="2023-12-31")
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [MagicMock(content=generated)]
    mock_agent.run = AsyncMock(return_value=mock_response)

    node = SearchNode(config={"use_llm": True})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)

    with patch("src.nodes.search_node.get_search_agent", return_value=mock_agent):
        input_data = SearchInput(
            query_keywords=["default"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索大模型在自动驾驶中的应用",
        )
        output = await node.process(input_data)

    assert output.total_count == 1
    assert output.metadata["querys"] == ["LLM", "autonomous driving"]
    assert output.metadata["start_date"] == "2023-01-01"
    assert output.metadata["end_date"] == "2023-12-31"
    node.paper_searcher.search_papers.assert_awaited_once_with(
        querys=["LLM", "autonomous driving"],
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )


@pytest.mark.asyncio
async def test_search_node_process_llm_fallback_to_keywords(sample_papers):
    node = SearchNode(config={"use_llm": True})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)

    with patch("src.nodes.search_node.get_search_agent", return_value=None):
        input_data = SearchInput(
            query_keywords=["fallback keyword"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索相关内容",
        )
        output = await node.process(input_data)

    assert output.total_count == 1
    assert output.metadata["querys"] == ["fallback keyword"]


@pytest.mark.asyncio
async def test_search_node_process_empty_keywords_raises():
    node = SearchNode(config={"use_llm": False})
    input_data = SearchInput(
        query_keywords=[],
        search_scope=SearchScope(max_results=5),
        user_request="",
    )
    with pytest.raises(ValueError, match="没有可用的搜索关键词"):
        await node.process(input_data)


@pytest.mark.asyncio
async def test_search_node_langgraph_adapter_success(sample_papers):
    current_state = PaperAgentState(
        user_request="ROS2 automated driving",
        max_papers=5,
        error=NodeError(),
        config={"search_node": {"use_llm": False}},
    )
    state: State = {"state_queue": asyncio.Queue(), "value": current_state}

    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(return_value=sample_papers)
    with patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher):
        result_state = await search_node(state)

    current = result_state["value"]
    assert current.search_output.total_count == 1
    assert current.error.search_node_error is None
    assert current.current_step == ExecutionState.SEARCHING


@pytest.mark.asyncio
async def test_search_node_langgraph_adapter_empty_results():
    current_state = PaperAgentState(
        user_request="nonexistent topic xyz",
        max_papers=5,
        error=NodeError(),
        config={"search_node": {"use_llm": False}},
    )
    state: State = {"state_queue": asyncio.Queue(), "value": current_state}

    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(return_value=[])
    with patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher):
        result_state = await search_node(state)

    current = result_state["value"]
    assert current.search_output.total_count == 0
    assert current.error.search_node_error == "没有找到相关论文,请尝试其他查询条件"


@pytest.mark.asyncio
async def test_search_node_langgraph_adapter_exception():
    current_state = PaperAgentState(
        user_request="ROS2",
        max_papers=5,
        error=NodeError(),
        config={"search_node": {"use_llm": False}},
    )
    state: State = {"state_queue": asyncio.Queue(), "value": current_state}

    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(side_effect=Exception("arxiv down"))
    with patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher):
        result_state = await search_node(state)

    current = result_state["value"]
    assert current.error.search_node_error.startswith("Search failed:")
    assert "arxiv down" in current.error.search_node_error


@pytest.mark.asyncio
async def test_search_node_langgraph_adapter_frontend_mode(sample_papers):
    """验证 frontend_enabled=True 时向 state_queue 推送 BackToFrontData。"""
    state_queue = asyncio.Queue()
    current_state = PaperAgentState(
        user_request="ROS2 automated driving",
        max_papers=5,
        error=NodeError(),
        config={"search_node": {"use_llm": False, "frontend_enabled": True}},
    )
    state: State = {"state_queue": state_queue, "value": current_state}

    mock_searcher = MagicMock()
    mock_searcher.search_papers = AsyncMock(return_value=sample_papers)
    with patch("src.nodes.search_node.PaperSearcher", return_value=mock_searcher):
        result_state = await search_node(state)

    current = result_state["value"]
    assert current.search_output.total_count == 1

    messages = []
    while not state_queue.empty():
        messages.append(state_queue.get_nowait())
    assert len(messages) == 2
    assert messages[0].state == "initializing"
    assert messages[1].state == "completed"
    assert "共找到 1 条结果" in messages[1].data


def test_parse_search_query():
    s = "querys = ['LLM', 'autonomous driving']\nstart_date = '2023-01-01'\nend_date = '2023-12-31'"
    query = parse_search_query(s)
    assert query.querys == ["LLM", "autonomous driving"]
    assert query.start_date == "2023-01-01"
    assert query.end_date == "2023-12-31"


def test_parse_search_query_invalid_list():
    s = "querys = [invalid\nstart_date = '2023-01-01'"
    query = parse_search_query(s)
    assert query.querys == []
    assert query.start_date == "2023-01-01"
    assert query.end_date is None


def test_parse_search_query_none_dates():
    """验证 LLM 输出 None 或空字符串时统一转为 None。"""
    s = "querys = ['LLM']\nstart_date = None\nend_date = ''"
    query = parse_search_query(s)
    assert query.querys == ["LLM"]
    assert query.start_date is None
    assert query.end_date is None


def test_search_node_normalize_and_validate_dates():
    """验证空字符串归一化与日期顺序自动修正。"""
    assert SearchNode._normalize_date("  ") is None
    assert SearchNode._normalize_date("none") is None
    assert SearchNode._normalize_date("2023-01-01") == "2023-01-01"

    start, end = SearchNode._validate_date_order("2023-12-31", "2023-01-01")
    assert start == "2023-01-01"
    assert end == "2023-12-31"


def test_to_search_result():
    node = SearchNode(config={"use_llm": False})
    paper = {
        "paper_id": "1234.56789",
        "title": "Test",
        "authors": ["A"],
        "summary": "S",
        "published": 2024,
        "published_date": "2024-01-01T00:00:00",
        "url": "http://arxiv.org/abs/1234.56789",
        "pdf_url": "http://arxiv.org/pdf/1234.56789",
    }
    result = node._to_search_result(paper)
    assert isinstance(result, SearchResult)
    assert result.paper_id == "1234.56789"
    assert result.published == "2024-01-01T00:00:00"
    assert result.metadata["published_year"] == 2024


@pytest.mark.asyncio
async def test_search_node_response_speed(sample_papers):
    import time

    node = SearchNode(config={"use_llm": False})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)
    input_data = SearchInput(
        query_keywords=["speed test"],
        search_scope=SearchScope(max_results=5),
        user_request="speed test request",
    )
    start = time.perf_counter()
    output = await node.process(input_data)
    elapsed = time.perf_counter() - start

    assert output.total_count == 1
    assert elapsed < 1.0, f"SearchNode.process took {elapsed:.3f}s, expected < 1s when mocked"
