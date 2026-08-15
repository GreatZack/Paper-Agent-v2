import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.core.model_client import create_model_client
from src.core.openai_client import SystemMessage, UserMessage
from src.core.prompts import parse_comparison_prompt, parse_taxonomy_prompt
from src.core.state_models import (
    BackToFrontData,
    ComparisonPoint,
    DiscrepancyNote,
    ExecutionState,
    KeyInformation,
    ParseInput,
    ParseOutput,
    ParsedPaper,
    ParseRule,
    SearchResult,
    State,
)
from src.nodes.base_node import BaseNode


class ParseNode(BaseNode[ParseInput, ParseOutput]):
    """解析节点：将多篇论文的 KeyInformation 清洗、聚合为结构化知识库。"""

    name = "parse_node"
    input_model = ParseInput
    output_model = ParseOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._model_client = None

    # ── 核心入口 ──────────────────────────────────────────────

    async def process(self, input_data: ParseInput) -> ParseOutput:
        raw_list = input_data.content

        if not raw_list:
            self.logger.info("[parse_node] 无关键信息，跳过处理")
            return ParseOutput(status="completed")

        self.logger.info(f"[parse_node] 开始处理 {len(raw_list)} 篇论文的关键信息")

        # ── Step 1: 规则清洗层 ──
        cleaned_papers: Dict[str, ParsedPaper] = {}
        raw_extractions: Dict[str, KeyInformation] = {}
        coverage: Dict[str, List[str]] = defaultdict(list)

        for kw in raw_list:
            paper_id = kw.paper_id
            meta = input_data.paper_meta.get(paper_id)
            cleaned = self._clean_single(kw, meta)
            cleaned_papers[paper_id] = cleaned
            raw_extractions[paper_id] = kw

            for dim in ("core_problem", "key_methodology", "main_results", "limitations"):
                val = getattr(kw, dim, "")
                if val and val not in ("未提及", ""):
                    coverage[dim].append(paper_id)

        self.logger.info(f"[parse_node] 规则清洗完成，有效论文={len(cleaned_papers)}")

        # ── Step 2a: LLM Taxonomy 分析 ──
        taxonomy_result = await self._call_llm_structured(
            system_prompt=parse_taxonomy_prompt,
            user_text=self._build_taxonomy_input(cleaned_papers),
            label="taxonomy",
        )
        if taxonomy_result:
            paper_tags = taxonomy_result.get("paper_tags", {})
            for pid, tags in paper_tags.items():
                if pid in cleaned_papers:
                    cleaned_papers[pid].tags = tags if isinstance(tags, list) else []
            taxonomy = taxonomy_result.get("taxonomy", {})
        else:
            taxonomy = {}

        self.logger.info(
            f"[parse_node] taxonomy 分析完成，类别数={len(taxonomy)}"
        )

        # ── Step 2b: LLM Comparison + Discrepancies 分析 ──
        comparison_result = await self._call_llm_structured(
            system_prompt=parse_comparison_prompt,
            user_text=self._build_comparison_input(cleaned_papers),
            label="comparison",
        )
        if comparison_result:
            comparison_points = [
                ComparisonPoint(**p)
                for p in comparison_result.get("comparison_points", [])
                if isinstance(p, dict)
            ]
            discrepancies = [
                DiscrepancyNote(**d)
                for d in comparison_result.get("discrepancies", [])
                if isinstance(d, dict)
            ]
        else:
            comparison_points = []
            discrepancies = []

        self.logger.info(
            f"[parse_node] 对比分析完成，对比点={len(comparison_points)} 矛盾={len(discrepancies)}"
        )

        # ── Step 3: 组装 ──
        return ParseOutput(
            papers=cleaned_papers,
            raw_extractions=raw_extractions,
            taxonomy=taxonomy,
            comparison_points=comparison_points,
            discrepancies=discrepancies,
            coverage=dict(coverage),
            status="completed",
        )

    # ── 清洗方法 ──────────────────────────────────────────────

    @staticmethod
    def _clean_single(kw: KeyInformation, meta: Optional[SearchResult]) -> ParsedPaper:
        """规则清洗单篇 KeyInformation → ParsedPaper。"""
        title = meta.title if meta and meta.title else ""
        authors_list = meta.authors if meta and meta.authors else []
        authors = ", ".join(authors_list) if authors_list else ""

        # 去空、去"未提及"
        core_problem = kw.core_problem if kw.core_problem and kw.core_problem != "未提及" else ""
        key_methodology = kw.key_methodology if kw.key_methodology and kw.key_methodology != "未提及" else ""
        main_results = kw.main_results if kw.main_results and kw.main_results != "未提及" else ""
        limitations = kw.limitations if kw.limitations and kw.limitations != "未提及" else ""

        # contributions：必须是 List[str]，去重
        contributions = list(dict.fromkeys(
            c.strip() for c in kw.contributions if c and c.strip() != "未提及"
        ))

        return ParsedPaper(
            paper_id=kw.paper_id,
            title=title,
            authors=authors,
            core_problem=core_problem,
            key_methodology=key_methodology,
            main_results=main_results,
            limitations=limitations,
            contributions=contributions,
        )

    # ── 输入文本构建 ──────────────────────────────────────────

    @staticmethod
    def _build_taxonomy_input(papers: Dict[str, ParsedPaper]) -> str:
        """为 taxonomy 分析构建输入（只送 title + key_methodology）。"""
        parts = []
        for pid, p in papers.items():
            parts.append(f"[{pid}] 标题: {p.title}\n方法描述: {p.key_methodology}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_comparison_input(papers: Dict[str, ParsedPaper]) -> str:
        """为对比分析构建输入（送 title + main_results + limitations）。"""
        parts = []
        for pid, p in papers.items():
            parts.append(
                f"[{pid}] 标题: {p.title}\n"
                f"主要结果: {p.main_results}\n"
                f"局限性: {p.limitations}"
            )
        return "\n\n".join(parts)

    # ── LLM 调用 ──────────────────────────────────────────────

    async def _call_llm_structured(
        self,
        system_prompt: str,
        user_text: str,
        label: str,
    ) -> Optional[Dict[str, Any]]:
        """调用 LLM 并解析 JSON 输出。"""
        if not user_text.strip():
            self.logger.warning(f"[parse_node] {label}: 输入为空，跳过")
            return None

        client = self._get_model_client()
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_text, source="user"),
        ]

        for attempt in range(2):
            try:
                result = await client.create(messages=messages, json_output=True)
                content = result.content

                # 清理 markdown 代码块包裹
                content = self._strip_json_fence(content)
                parsed = json.loads(content)
                self.logger.info(f"[parse_node] {label}: 分析成功")
                return parsed

            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.warning(
                    f"[parse_node] {label}: 第 {attempt + 1} 次解析失败 — {e}"
                )
                if attempt == 0:
                    continue
                self.logger.error(f"[parse_node] {label}: 重试后仍失败")
                return None
            except Exception as e:
                self.logger.error(f"[parse_node] {label}: LLM 调用异常 — {e}")
                return None

        return None

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        """去除 ```json ... ``` 包裹。"""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return text

    # ── 模型客户端 ────────────────────────────────────────────

    def _get_model_client(self):
        """懒加载 LLM 客户端。"""
        if self._model_client is None:
            self._model_client = create_model_client()
        return self._model_client


# ── LangGraph 适配函数 ────────────────────────────────────────


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

        # 从 search_output 构建 paper_meta
        search_results = {
            r.paper_id: r for r in current_state.search_output.results
        }
        parse_input = ParseInput(
            content=current_state.read_output.key_info,
            paper_meta=search_results,
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
                data=f"解析完成，论文数={len(parse_output.papers)} "
                     f"分类数={len(parse_output.taxonomy)} "
                     f"对比点={len(parse_output.comparison_points)}",
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
