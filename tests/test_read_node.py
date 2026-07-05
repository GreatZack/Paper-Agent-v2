"""read_node 测试：离线单元 + 端到端单篇。"""

import asyncio
import json
import os
import pytest

from src.core.state_models import (
    KeyInformation,
    ReadInput,
    ReadOutput,
    ReadingStrategy,
    SearchResult,
)
from src.nodes.read_node import ReadNode


# ── fixture ────────────────────────────────────────────────────

@pytest.fixture
def node():
    """默认配置的 ReadNode。"""
    return ReadNode({"concurrency": 2, "max_tokens_threshold": 90000})


@pytest.fixture
def sample_search_result():
    """取一个真实 PDF 构造 SearchResult。"""
    papers_dir = "data/papers"
    pdfs = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    if not pdfs:
        pytest.skip("data/papers/ 中没有 PDF 文件")
    pdf_path = os.path.join(papers_dir, pdfs[0])
    paper_id = os.path.splitext(pdfs[0])[0]
    return SearchResult(
        paper_id=paper_id,
        title="Test Paper",
        authors=["Author A", "Author B"],
        summary="Test abstract.",
        pdf_path=pdf_path,
    )


# ── 离线单元测试（不调 API）────────────────────────────────────

class TestTokenEstimation:
    def test_english_text(self, node):
        assert node._estimate_tokens("hello world") == 2

    def test_chinese_text(self, node):
        # 每个中文字符 ~1 token（粗略按 4 字节/字估算）
        assert node._estimate_tokens("你好世界") == 1

    def test_long_text(self, node):
        text = "x" * 4000
        assert node._estimate_tokens(text) == 1000


class TestSplitBySection:
    def test_basic_split(self, node):
        md = "# Title\n\n## Abstract\ntext A\n\n## Introduction\ntext B"
        chunks = node._split_by_section(md)
        assert len(chunks) == 2
        assert "Abstract" in chunks[0]
        assert "Introduction" in chunks[1]

    def test_no_sections(self, node):
        md = "# Title\n\nPlain text without sections."
        chunks = node._split_by_section(md)
        assert len(chunks) == 1
        assert "Plain text" in chunks[0]


class TestStripJsonFence:
    def test_with_fence(self, node):
        raw = '```json\n{"key": "value"}\n```'
        assert node._strip_json_fence(raw) == '{"key": "value"}'

    def test_without_fence(self, node):
        raw = '{"key": "value"}'
        assert node._strip_json_fence(raw) == '{"key": "value"}'

    def test_strips_whitespace(self, node):
        raw = '  {"key": "value"}  '
        assert node._strip_json_fence(raw) == '{"key": "value"}'


class TestMergeChunkResults:
    def test_single_chunk(self, node):
        k = KeyInformation(
            paper_id="p1", core_problem="test",
            evidence_sections={"core_problem": "§1"},
        )
        result = node._merge_chunk_results([k])
        assert result.core_problem == "test"

    def test_merge_prefers_abstract_for_core_problem(self, node):
        k1 = KeyInformation(
            paper_id="p1", core_problem="",
            evidence_sections={"core_problem": "Introduction §1"},
        )
        k2 = KeyInformation(
            paper_id="p1", core_problem="real problem",
            evidence_sections={"core_problem": "Abstract"},
        )
        result = node._merge_chunk_results([k1, k2])
        assert result.core_problem == "real problem"

    def test_merge_concat_results(self, node):
        k1 = KeyInformation(
            paper_id="p1", main_results="result A",
        )
        k2 = KeyInformation(
            paper_id="p1", main_results="result B",
        )
        result = node._merge_chunk_results([k1, k2])
        assert "result A" in result.main_results
        assert "result B" in result.main_results

    def test_contributions_dedup(self, node):
        k1 = KeyInformation(paper_id="p1", contributions=["A", "B"])
        k2 = KeyInformation(paper_id="p1", contributions=["B", "C"])
        result = node._merge_chunk_results([k1, k2])
        assert result.contributions == ["A", "B", "C"]


# ── PDF 提取测试（调 pymupdf4llm，不调 LLM）──────────────────

class TestExtractMarkdown:
    def test_extracts_valid_markdown(self, node, sample_search_result):
        md = node._extract_markdown(sample_search_result.pdf_path)
        assert len(md) > 500
        assert "# " in md  # 至少有标题

    def test_missing_pdf_raises(self, node):
        with pytest.raises(Exception):
            node._extract_markdown("data/papers/nonexistent.pdf")


# ── 空文档处理 ─────────────────────────────────────────────────

class TestEmptyDocuments:
    def test_empty_documents_returns_empty_output(self, node):
        result = asyncio.run(node.process(
            ReadInput(documents=[], reading_strategy=ReadingStrategy())
        ))
        assert isinstance(result, ReadOutput)
        assert result.key_info == []
        assert result.status == "completed"


# ── 失败路径（无 pdf_path）────────────────────────────────────

class TestFailedPapers:
    def test_no_pdf_path_goes_to_failed(self, node):
        paper = SearchResult(
            paper_id="no_pdf", title="No PDF", pdf_path=None,
        )
        result = asyncio.run(node.process(
            ReadInput(documents=[paper], reading_strategy=ReadingStrategy())
        ))
        assert "no_pdf" in result.failed_paper_ids
        assert len(result.key_info) == 0


# ── 端到端单篇测试（真调 DeepSeek API）─────────────────────────

@pytest.mark.slow
class TestEndToEnd:
    def test_single_paper_extraction(self, node, sample_search_result):
        """用真实 LLM 提取一篇论文的关键信息。"""
        strategy = ReadingStrategy(
            focus_areas=["method", "result", "limitation"],
            depth="medium",
        )
        result = asyncio.run(node._process_single(sample_search_result, strategy))

        assert result is not None, "单篇提取不应返回 None"
        assert result.paper_id != ""
        assert result.core_problem != "", "core_problem 不应为空"
        assert result.key_methodology != "", "key_methodology 不应为空"
        print(f"\n core_problem: {result.core_problem}")
        print(f" key_methodology: {result.key_methodology[:200]}...")
        print(f" main_results: {result.main_results[:200]}...")
        print(f" evidence_sections: {result.evidence_sections}")
