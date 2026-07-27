import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import arxiv
import pytest

from src.core.state_models import (
    ExecutionState,
    NodeError,
    PaperAgentState,
    SearchInput,
    SearchOutput,
    SearchResult,
    SearchScope,
    State,
)
from src.nodes.search_node import SearchNode, search_node


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
    node._filter_relevant_papers = AsyncMock(side_effect=lambda papers, *a, **kw: (papers, []))

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
        query='(all:"ROS2" AND all:"automated driving")',
        max_results=25,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )


@pytest.mark.asyncio
async def test_search_node_process_with_llm_generated_query(sample_papers):
    """LLM 返回完整查询表达式，验证透传。"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [MagicMock(content='(all:"LLM" AND all:"autonomous driving") AND submittedDate:[20230101 TO 20231231]')]
    mock_agent.run = AsyncMock(return_value=mock_response)

    node = SearchNode(config={"use_llm": True})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)
    node._filter_relevant_papers = AsyncMock(side_effect=lambda papers, *a, **kw: (papers, []))

    with patch.object(node, "_get_search_agent", return_value=mock_agent):
        input_data = SearchInput(
            query_keywords=["default"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索大模型在自动驾驶中的应用",
        )
        output = await node.process(input_data)

    assert output.total_count == 1
    # 验证 metadata 中记录了 LLM 生成的完整查询
    assert 'all:"LLM"' in output.metadata["arxiv_query"]
    assert 'all:"autonomous driving"' in output.metadata["arxiv_query"]
    node.paper_searcher.search_papers.assert_awaited_once_with(
        query='(all:"LLM" AND all:"autonomous driving") AND submittedDate:[20230101 TO 20231231]',
        max_results=25,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )


@pytest.mark.asyncio
async def test_search_node_with_scope_date_uses_relevance_sort(sample_papers):
    """SearchScope 中指定了 start_date 时，排序使用 Relevance。"""
    node = SearchNode(config={"use_llm": False})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)
    node._filter_relevant_papers = AsyncMock(side_effect=lambda papers, *a, **kw: (papers, []))

    input_data = SearchInput(
        query_keywords=["LLM"],
        search_scope=SearchScope(max_results=5, start_date="2023-01-01"),
        user_request="搜索大模型",
    )
    await node.process(input_data)

    node.paper_searcher.search_papers.assert_awaited_once_with(
        query='(all:"LLM")',
        max_results=25,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )


@pytest.mark.asyncio
async def test_search_node_process_llm_agent_unavailable_raises(sample_papers):
    """LLM Agent 不可用时，重试耗尽后报错（不做降级）。"""
    node = SearchNode(config={"use_llm": True, "query_retry_limit": 2})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)

    with patch.object(node, "_get_search_agent", return_value=None):
        input_data = SearchInput(
            query_keywords=["fallback keyword"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索相关内容",
        )
        with pytest.raises(ValueError, match="LLM 搜索 Agent 初始化失败"):
            await node.process(input_data)


@pytest.mark.asyncio
async def test_search_node_process_llm_invalid_query_all_retries_exhausted(sample_papers):
    """LLM 持续返回无效查询时，重试耗尽后报错。"""
    mock_agent = MagicMock()
    mock_response = MagicMock()
    mock_response.messages = [MagicMock(content="invalid query (unbalanced")]  # 括号不平衡
    mock_agent.run = AsyncMock(return_value=mock_response)

    node = SearchNode(config={"use_llm": True, "query_retry_limit": 2})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)

    with patch.object(node, "_get_search_agent", return_value=mock_agent):
        input_data = SearchInput(
            query_keywords=["fallback"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索相关内容",
        )
        with pytest.raises(ValueError, match="LLM 无法生成有效的查询表达式"):
            await node.process(input_data)

    # 验证 LLM 确实被调用了 query_retry_limit 次
    assert mock_agent.run.await_count == 2


@pytest.mark.asyncio
async def test_search_node_process_llm_retry_success(sample_papers):
    """LLM 前几次返回无效查询，最后一次正常，验证重试机制生效。"""
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=[
        MagicMock(messages=[MagicMock(content="bad query (unbalanced")]),
        MagicMock(messages=[MagicMock(content="another bad (query")]),
        MagicMock(messages=[MagicMock(content='(all:"LLM" AND all:"autonomous driving")')]),
    ])

    node = SearchNode(config={"use_llm": True, "query_retry_limit": 3})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)
    node._filter_relevant_papers = AsyncMock(side_effect=lambda papers, *a, **kw: (papers, []))

    with patch.object(node, "_get_search_agent", return_value=mock_agent):
        input_data = SearchInput(
            query_keywords=["default"],
            search_scope=SearchScope(max_results=5),
            user_request="帮我搜索大模型在自动驾驶中的应用",
        )
        output = await node.process(input_data)

    assert output.total_count == 1
    assert 'all:"LLM"' in output.metadata["arxiv_query"]
    assert mock_agent.run.await_count == 3  # 前 2 次失败，第 3 次成功


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


def test_validate_query():
    """验证查询表达式校验函数。"""
    # 合法查询
    assert SearchNode._validate_query('(all:"LLM" AND all:"autonomous driving")') is True
    assert SearchNode._validate_query('ti:"GPT" OR abs:"self-driving"') is True
    assert SearchNode._validate_query('(all:"A" OR all:"B") AND (all:"C" OR all:"D")') is True

    # 非法查询
    assert SearchNode._validate_query('') is False
    assert SearchNode._validate_query('  ') is False
    assert SearchNode._validate_query('((unbalanced') is False  # 括号不平衡
    assert SearchNode._validate_query('unbalanced)') is False  # 括号负深度
    assert SearchNode._validate_query('unknown_field:test') is False  # 不允许的字段


@pytest.mark.asyncio
async def test_filter_relevant_papers_returns_discarded():
    """验证 _filter_relevant_papers 返回 (kept, discarded)，discarded 包含被丢弃论文信息。"""
    papers = [
        {"paper_id": "001", "title": "Paper A", "summary": "A"},
        {"paper_id": "002", "title": "Paper B", "summary": "B"},
        {"paper_id": "003", "title": "Paper C", "summary": "C"},
        {"paper_id": "004", "title": "Paper D", "summary": "D"},
        {"paper_id": "005", "title": "Paper E", "summary": "E"},
    ]

    node = SearchNode(config={"use_llm": False})

    mock_response = AsyncMock()
    mock_response.messages = [MagicMock(content="[0, 2, 4]")]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_response)

    with patch("src.nodes.search_node.AssistantAgent", return_value=mock_agent):
        with patch("src.nodes.search_node.create_model_client") as mock_client:
            mock_client.return_value = MagicMock(close=AsyncMock())
            kept, discarded = await node._filter_relevant_papers(
                papers, "test request", ["test"]
            )

    assert len(kept) == 3
    assert kept[0]["paper_id"] == "001"
    assert kept[1]["paper_id"] == "003"
    assert kept[2]["paper_id"] == "005"
    assert len(discarded) == 2
    assert discarded[0] == {"title": "Paper B", "paper_id": "002", "index": 1}
    assert discarded[1] == {"title": "Paper D", "paper_id": "004", "index": 3}


@pytest.mark.asyncio
async def test_filter_relevant_papers_discarded_empty_when_parse_fails():
    """LLM 返回无法解析的内容时，全部保留，discarded 为空。"""
    papers = [
        {"paper_id": "001", "title": "Paper A", "summary": "A"},
        {"paper_id": "002", "title": "Paper B", "summary": "B"},
    ]

    node = SearchNode(config={"use_llm": False})

    mock_response = AsyncMock()
    mock_response.messages = [MagicMock(content="无法解析的格式")]
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_response)

    with patch("src.nodes.search_node.AssistantAgent", return_value=mock_agent):
        with patch("src.nodes.search_node.create_model_client") as mock_client:
            mock_client.return_value = MagicMock(close=AsyncMock())
            kept, discarded = await node._filter_relevant_papers(
                papers, "test request", ["test"]
            )

    assert len(kept) == 2  # 全部保留
    assert discarded == []  # 丢弃记录为空


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


@pytest.mark.parametrize(
    ("target_count", "expected_candidates"),
    [(1, 20), (3, 20), (5, 25), (8, 40), (10, 50)],
)
def test_candidate_pool_size(target_count, expected_candidates):
    node = SearchNode(config={"use_llm": False})

    assert node._candidate_count(target_count) == expected_candidates


@pytest.mark.asyncio
async def test_search_filters_candidates_before_applying_target_limit(sample_papers):
    candidates = [
        {**sample_papers[0], "paper_id": f"paper-{index}"} for index in range(20)
    ]
    node = SearchNode(config={"use_llm": False})
    node.paper_searcher.search_papers = AsyncMock(return_value=candidates)
    node._filter_relevant_papers = AsyncMock(
        side_effect=lambda papers, *args: (papers, [])
    )

    output = await node.process(
        SearchInput(
            query_keywords=["transformer"],
            search_scope=SearchScope(max_results=1),
            user_request="transformer architecture",
        )
    )

    assert output.total_count == 1
    assert output.metadata["candidate_count"] == 20
    assert output.metadata["filtered_kept"] == 20
    assert output.metadata["returned_count"] == 1
    node.paper_searcher.search_papers.assert_awaited_once_with(
        query='(all:"transformer")',
        max_results=20,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )


def test_explicit_latest_request_uses_submitted_date_sort():
    node = SearchNode(config={"use_llm": False})

    assert (
        node._sort_criterion("查找最新的 Transformer 论文")
        == arxiv.SortCriterion.SubmittedDate
    )


def test_search_agents_are_isolated_per_node():
    clients = [MagicMock(), MagicMock()]
    agents = [MagicMock(), MagicMock()]

    with (
        patch("src.nodes.search_node.create_model_client", side_effect=clients),
        patch("src.nodes.search_node.AssistantAgent", side_effect=agents),
    ):
        first = SearchNode()._get_search_agent()
        second = SearchNode()._get_search_agent()

    assert first is agents[0]
    assert second is agents[1]
    assert first is not second


@pytest.mark.asyncio
async def test_search_node_response_speed(sample_papers):
    import time

    node = SearchNode(config={"use_llm": False})
    node.paper_searcher.search_papers = AsyncMock(return_value=sample_papers)
    node._filter_relevant_papers = AsyncMock(side_effect=lambda papers, *a, **kw: (papers, []))
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
