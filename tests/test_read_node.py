"""read_node 测试：离线单元 + 端到端单篇。"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import pytest

from src.core.state_models import (
    KeyInformation,
    ReadingStrategy,
    ReadInput,
    ReadOutput,
    SearchResult,
)
from src.nodes.read_node import ReadNode

# ── fixture ────────────────────────────────────────────────────


@pytest.fixture
def node():
    """默认配置的 ReadNode。"""
    return ReadNode(
        {
            "concurrency": 2,
            "max_tokens_threshold": 90000,
        }
    )


@pytest.fixture
def sample_search_result():
    """取一个真实 PDF 构造 SearchResult。"""
    papers_dir = "data/papers"
    if not os.path.isdir(papers_dir):
        pytest.skip("data/papers/ 目录不存在")
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

    def test_model_chunks_stay_within_threshold(self):
        node = ReadNode({"max_tokens_threshold": 10})
        md = "# Title\n\n## A\n" + "a" * 60 + "\n\n## B\n" + "b" * 60

        chunks = node._chunk_text_for_model(md)

        assert len(chunks) > 1
        assert all(node._estimate_tokens(chunk) <= 10 for chunk in chunks)


class TestAuthorFormatting:
    def test_truncates_very_long_author_list(self, node):
        authors = [f"Author {i}" for i in range(25)]

        result = node._format_authors(authors)

        assert "Author 19" in result
        assert "Author 20" not in result
        assert "另有 5 位作者" in result


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
            paper_id="p1",
            core_problem="test",
            evidence_sections={"core_problem": "§1"},
        )
        result = node._merge_chunk_results([k])
        assert result.core_problem == "test"

    def test_merge_prefers_abstract_for_core_problem(self, node):
        k1 = KeyInformation(
            paper_id="p1",
            core_problem="",
            evidence_sections={"core_problem": "Introduction §1"},
        )
        k2 = KeyInformation(
            paper_id="p1",
            core_problem="real problem",
            evidence_sections={"core_problem": "Abstract"},
        )
        result = node._merge_chunk_results([k1, k2])
        assert result.core_problem == "real problem"

    def test_merge_concat_results(self, node):
        k1 = KeyInformation(
            paper_id="p1",
            main_results="result A",
        )
        k2 = KeyInformation(
            paper_id="p1",
            main_results="result B",
        )
        result = node._merge_chunk_results([k1, k2])
        assert "result A" in result.main_results
        assert "result B" in result.main_results

    def test_contributions_dedup(self, node):
        k1 = KeyInformation(paper_id="p1", contributions=["A", "B"])
        k2 = KeyInformation(paper_id="p1", contributions=["B", "C"])
        result = node._merge_chunk_results([k1, k2])
        assert result.contributions == ["A", "B", "C"]

    def test_merge_preserves_unmentioned_limitations(self, node):
        chunks = [
            KeyInformation(paper_id="p1", limitations="未提及"),
            KeyInformation(paper_id="p1", limitations=""),
        ]

        result = node._merge_chunk_results(chunks)

        assert result.limitations == "未提及"

    def test_real_limitation_takes_precedence_over_unmentioned(self, node):
        chunks = [
            KeyInformation(paper_id="p1", limitations="未提及"),
            KeyInformation(paper_id="p1", limitations="上下文长度仍有限"),
        ]

        result = node._merge_chunk_results(chunks)

        assert result.limitations == "上下文长度仍有限"


class TestVerification:
    @pytest.mark.asyncio
    async def test_optional_empty_field_is_verified_as_unmentioned(self, node):
        node._model_client = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "limitations": {
                                "verified": True,
                                "exact_quote": "原文未明确列出局限性",
                                "reason": "",
                            }
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        key_info = KeyInformation(paper_id="p1", limitations="")

        result = await node._verify_one_paper(
            "论文正文没有局限性章节",
            key_info,
            fields_to_verify=["limitations"],
            optional_fields=["limitations"],
        )

        assert result is not None
        assert result.passed is True
        assert key_info.limitations == "未提及"
        assert [item.field for item in result.items] == ["limitations"]

    @pytest.mark.asyncio
    async def test_empty_verification_field_list_skips_model_call(self, node):
        model_client = SimpleNamespace(create=AsyncMock())
        node._model_client = model_client

        result = await node._verify_one_paper(
            "论文正文",
            KeyInformation(paper_id="p1"),
            fields_to_verify=["unsupported_field"],
        )

        assert result is not None
        assert result.passed is True
        model_client.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_optional_verification_failure_does_not_block_paper(self, node):
        node._model_client = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "key_methodology": {
                                "verified": True,
                                "exact_quote": "method evidence",
                                "reason": "",
                            },
                            "limitations": {
                                "verified": False,
                                "exact_quote": "",
                                "reason": "原文没有支持该局限性描述",
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )
        key_info = KeyInformation(
            paper_id="p1",
            key_methodology="可靠的方法描述",
            limitations="无法证实的局限性",
        )

        result = await node._verify_one_paper(
            "论文正文",
            key_info,
            fields_to_verify=["key_methodology", "limitations"],
            optional_fields=["limitations"],
        )

        assert result is not None
        assert result.passed is True
        assert key_info.limitations == "未能从原文可靠验证"
        assert result.items[1].verified is False

    @pytest.mark.asyncio
    async def test_required_verification_failure_blocks_paper(self, node):
        node._model_client = SimpleNamespace(
            create=AsyncMock(
                return_value=SimpleNamespace(
                    content=json.dumps(
                        {
                            "main_results": {
                                "verified": False,
                                "exact_quote": "",
                                "reason": "结果数值与原文不一致",
                            }
                        },
                        ensure_ascii=False,
                    )
                )
            )
        )

        result = await node._verify_one_paper(
            "论文正文",
            KeyInformation(paper_id="p1", main_results="错误结果"),
            fields_to_verify=["main_results"],
            optional_fields=["limitations"],
        )

        assert result is not None
        assert result.passed is False


# ── PDF 提取测试（纯文本模式，不调 LLM）─────────────────────


class TestExtractMarkdown:
    def test_plain_text_mode_extracts_text(self, node, tmp_path):
        pdf_path = tmp_path / "plain-text.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "Lightweight PDF extraction")
            doc.save(pdf_path)

        text, pages = node._extract_pdf_content(str(pdf_path))

        assert "Lightweight PDF extraction" in text
        assert len(pages) == 1

    def test_extracts_valid_text(self, node, sample_search_result):
        md, pages = node._extract_pdf_content(sample_search_result.pdf_path)
        assert len(md) > 500
        assert pages
        assert any(page.strip() for page in pages)

    def test_missing_pdf_raises(self, node):
        with pytest.raises(Exception):
            node._extract_pdf_content("data/papers/nonexistent.pdf")


# ── 空文档处理 ─────────────────────────────────────────────────


class TestEmptyDocuments:
    def test_empty_documents_returns_empty_output(self, node):
        result = asyncio.run(
            node.process(ReadInput(documents=[], reading_strategy=ReadingStrategy()))
        )
        assert isinstance(result, ReadOutput)
        assert result.key_info == []
        assert result.status == "completed"


# ── 失败路径（无 pdf_path）────────────────────────────────────


class TestFailedPapers:
    def test_no_pdf_path_goes_to_failed(self, node):
        paper = SearchResult(
            paper_id="no_pdf",
            title="No PDF",
            pdf_path=None,
        )
        result = asyncio.run(
            node.process(
                ReadInput(documents=[paper], reading_strategy=ReadingStrategy())
            )
        )
        assert "no_pdf" in result.failed_paper_ids
        assert len(result.key_info) == 0


# ── 端到端单篇测试（真调 Mimo API）─────────────────────────


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
