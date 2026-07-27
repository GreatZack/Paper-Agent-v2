import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from autogen_core.models import SystemMessage, UserMessage

from src.core.model_client import create_model_client
from src.core.prompts import (
    write_batch_summary_prompt,
    write_consolidation_prompt,
    write_report_prompt,
)
from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    State,
    WriteInput,
    WriteOutput,
)
from src.nodes.base_node import BaseNode


class WriteNode(BaseNode[WriteInput, WriteOutput]):
    """撰写节点：基于结构化数据生成综述报告。"""

    name = "write_node"
    input_model = WriteInput
    output_model = WriteOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._model_client = None
        self._token_budget = int(self.config.get("token_budget", 64000))
        self._budget_ratio = float(self.config.get("budget_ratio", 0.6))
        self._batch_size = int(self.config.get("batch_size", 10))

    async def process(self, input_data: WriteInput) -> WriteOutput:
        papers = input_data.structured_data.get("papers", {})
        if not papers:
            self.logger.info("[write_node] 无论文，跳过撰写")
            return WriteOutput(generated_text="", status="completed", section_map={})

        estimated_tokens = self._estimate_input_tokens(input_data)
        self.logger.info(
            f"[write_node] estimated_input_tokens={estimated_tokens}, "
            f"budget={self._token_budget}, ratio={self._budget_ratio}"
        )

        if estimated_tokens <= self._token_budget * self._budget_ratio:
            self.logger.info("[write_node] strategy=single_pass")
            return await self._process_single_pass(input_data)
        else:
            self.logger.info("[write_node] strategy=multi_phase")
            return await self._process_multi_phase(input_data)

    def _estimate_input_tokens(self, input_data: WriteInput) -> int:
        total_chars = len(json.dumps(input_data.structured_data, ensure_ascii=False))
        return total_chars // 3

    async def _process_single_pass(self, input_data: WriteInput) -> WriteOutput:
        user_text = self._build_report_input(input_data.structured_data)
        result = await self._call_llm(write_report_prompt, user_text)

        if result is None:
            self.logger.error("[write_node] single_pass LLM failed after retries, using emergency output")
            return self._emergency_output(
                input_data.structured_data.get("papers", {}),
                input_data.structured_data.get("taxonomy"),
            )

        cleaned_text, section_map = self._parse_markdown_sections(result)
        return WriteOutput(generated_text=cleaned_text, status="completed", section_map=section_map)

    async def _process_multi_phase(self, input_data: WriteInput) -> WriteOutput:
        structured = input_data.structured_data
        papers = structured.get("papers", {})
        taxonomy = structured.get("taxonomy", {})
        comparison_points = structured.get("comparison_points", [])
        discrepancies = structured.get("discrepancies", [])
        coverage = structured.get("coverage", {})

        paper_ids = list(papers.keys())
        batches = self._split_into_batches(paper_ids, self._batch_size)
        self.logger.info(
            f"[write_node] multi_phase: {len(paper_ids)} papers, "
            f"{len(batches)} batches (batch_size={self._batch_size})"
        )

        tasks = []
        for batch_ids in batches:
            batch_papers = {pid: papers[pid] for pid in batch_ids}
            tasks.append(self._generate_batch_summary(batch_papers))

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_summaries = []
        fail_count = 0
        for r in batch_results:
            if isinstance(r, Exception):
                fail_count += 1
                self.logger.error(f"[write_node] batch failed: {r}")
            else:
                successful_summaries.append(r)

        total = len(batches)
        if total > 0 and fail_count / total > 0.5:
            self.logger.warning(
                f"[write_node] >50% batches failed ({fail_count}/{total}), "
                f"degrading to single_pass"
            )
            return await self._process_single_pass(input_data)

        consolidation_text = self._build_consolidation_input(
            successful_summaries, taxonomy, comparison_points, discrepancies, coverage
        )
        result = await self._call_llm(write_consolidation_prompt, consolidation_text)

        if result is None:
            self.logger.error("[write_node] consolidation LLM failed, using emergency output")
            return self._emergency_output(papers, taxonomy)

        cleaned_text, section_map = self._parse_markdown_sections(result)
        return WriteOutput(generated_text=cleaned_text, status="completed", section_map=section_map)

    async def _generate_batch_summary(self, batch_papers: Dict[str, Any]) -> Dict:
        user_text = self._build_batch_summary_input(batch_papers)
        result = await self._call_llm_json(
            write_batch_summary_prompt, user_text, label="batch_summary"
        )
        if result is None:
            raise RuntimeError(
                f"batch summary generation failed for {len(batch_papers)} papers"
            )
        return result

    async def _call_llm(self, system_prompt: str, user_text: str) -> Optional[str]:
        if not user_text.strip():
            return None
        client = self._get_model_client()
        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_text, source="user"),
        ]
        for attempt in range(2):
            try:
                result = await client.create(messages=messages, json_output=False)
                content = result.content
                if isinstance(content, list):
                    content = " ".join(str(t) for t in content)
                return str(content).strip()
            except Exception as e:
                self.logger.warning(f"[write_node] LLM call attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    continue
                return None
        return None

    async def _call_llm_json(
        self, system_prompt: str, user_text: str, label: str = ""
    ) -> Optional[Dict]:
        if not user_text.strip():
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
                content = self._strip_json_fence(str(content))
                return json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"[write_node] [{label}] attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    continue
                return None
        return None

    @staticmethod
    def _parse_markdown_sections(text: str) -> Tuple[str, Dict[str, str]]:
        section_map = {}
        current_section = None
        current_lines = []

        for line in text.split("\n"):
            anchor_match = re.match(r'<!--\s*section=(\w+)\s*-->', line.strip())
            if anchor_match:
                if current_section is not None:
                    section_map[current_section] = "\n".join(current_lines).strip()
                current_section = anchor_match.group(1)
                current_lines = []
                continue
            if current_section is not None:
                current_lines.append(line)

        if current_section is not None:
            section_map[current_section] = "\n".join(current_lines).strip()

        cleaned_text = re.sub(r'<!--\s*section=\w+\s*-->\n?', '', text).strip()
        return cleaned_text, section_map

    @staticmethod
    def _build_report_input(structured: Dict[str, Any]) -> str:
        parts = []
        papers = structured.get("papers", {})
        taxonomy = structured.get("taxonomy", {})
        comparison_points = structured.get("comparison_points", [])
        discrepancies = structured.get("discrepancies", [])
        coverage = structured.get("coverage", {})

        parts.append("## 论文列表")
        for pid, paper in papers.items():
            parts.append(f"[{pid}] 标题: {paper.get('title', '')}")
            parts.append(f"作者: {paper.get('authors', '')}")
            parts.append(f"核心问题: {paper.get('core_problem', '')}")
            parts.append(f"关键方法: {paper.get('key_methodology', '')}")
            parts.append(f"主要结果: {paper.get('main_results', '')}")
            parts.append(f"局限性: {paper.get('limitations', '')}")
            parts.append("")

        parts.append("## 论文分类")
        parts.append(json.dumps(taxonomy, ensure_ascii=False, indent=2))
        parts.append("")

        if comparison_points:
            parts.append("## 可对比数据点")
            for cp in comparison_points:
                parts.append(json.dumps(cp if isinstance(cp, dict) else cp.model_dump(), ensure_ascii=False))
            parts.append("")

        if discrepancies:
            parts.append("## 矛盾点")
            for d in discrepancies:
                parts.append(json.dumps(d if isinstance(d, dict) else d.model_dump(), ensure_ascii=False))
            parts.append("")

        parts.append("## 覆盖维度")
        parts.append(json.dumps(coverage, ensure_ascii=False, indent=2))
        return "\n".join(parts)

    @staticmethod
    def _build_batch_summary_input(batch_papers: Dict[str, Any]) -> str:
        parts = []
        for pid, paper in batch_papers.items():
            parts.append(f"[{pid}] 标题: {paper.get('title', '')}")
            parts.append(f"作者: {paper.get('authors', '')}")
            parts.append(f"核心问题: {paper.get('core_problem', '')}")
            parts.append(f"关键方法: {paper.get('key_methodology', '')}")
            parts.append(f"主要结果: {paper.get('main_results', '')}")
            parts.append(f"局限性: {paper.get('limitations', '')}")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _build_consolidation_input(
        summaries: List[Dict],
        taxonomy: Dict,
        comparison_points: List,
        discrepancies: List,
        coverage: Dict,
    ) -> str:
        parts = []
        parts.append("## 各批次论文摘要")
        for i, summary in enumerate(summaries):
            parts.append(f"### 批次 {i+1}")
            parts.append(json.dumps(summary, ensure_ascii=False, indent=2))
            parts.append("")

        parts.append("## 论文分类")
        parts.append(json.dumps(taxonomy, ensure_ascii=False, indent=2))
        parts.append("")

        if comparison_points:
            parts.append("## 可对比数据点")
            for cp in comparison_points:
                parts.append(json.dumps(cp if isinstance(cp, dict) else cp.model_dump(), ensure_ascii=False))
            parts.append("")

        if discrepancies:
            parts.append("## 矛盾点")
            for d in discrepancies:
                parts.append(json.dumps(d if isinstance(d, dict) else d.model_dump(), ensure_ascii=False))
            parts.append("")

        parts.append("## 覆盖维度")
        parts.append(json.dumps(coverage, ensure_ascii=False, indent=2))
        return "\n".join(parts)

    @staticmethod
    def _emergency_output(papers: Dict[str, Any], taxonomy: Optional[Dict] = None) -> WriteOutput:
        lines = ["# 综述报告（自动生成 - 简化版）", ""]
        if taxonomy:
            lines.append("## 论文分类")
            for cat, pids in taxonomy.items():
                lines.append(f"### {cat}")
                for pid in pids:
                    paper = papers.get(pid, {})
                    lines.append(f"- {paper.get('title', pid)}")
                lines.append("")
        for pid, paper in papers.items():
            lines.append(f"### {paper.get('title', pid)}")
            lines.append(f"**作者**：{paper.get('authors', '')}")
            if paper.get("core_problem"):
                lines.append(f"**核心问题**：{paper['core_problem']}")
            if paper.get("key_methodology"):
                lines.append(f"**关键方法**：{paper['key_methodology']}")
            if paper.get("main_results"):
                lines.append(f"**主要结果**：{paper['main_results']}")
            if paper.get("limitations"):
                lines.append(f"**局限性**：{paper['limitations']}")
            lines.append("")
        return WriteOutput(generated_text="\n".join(lines), status="completed", section_map={})

    @staticmethod
    def _split_into_batches(items: List[str], batch_size: int) -> List[List[str]]:
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return text

    def _get_model_client(self):
        if self._model_client is None:
            self._model_client = create_model_client()
        return self._model_client


async def write_node(state: State) -> State:
    """LangGraph 适配函数：撰写节点。"""
    state_queue = state["state_queue"]
    current_state = state["value"]

    try:
        current_state.current_step = ExecutionState.WRITING
        await state_queue.put(
            BackToFrontData(step=ExecutionState.WRITING, state="initializing", data=None)
        )

        node = WriteNode(current_state.config.get("write_node", {}))

        write_input = WriteInput(
            structured_data={
                "papers": {pid: p.model_dump() for pid, p in current_state.parse_output.papers.items()},
                "taxonomy": current_state.parse_output.taxonomy,
                "comparison_points": [p.model_dump() for p in current_state.parse_output.comparison_points],
                "discrepancies": [d.model_dump() for d in current_state.parse_output.discrepancies],
                "coverage": current_state.parse_output.coverage,
            },
            template=current_state.config.get("write_template", ""),
            style_hints={"language": "zh", "tone": "academic"},
        )

        write_output = await node.run(write_input)
        current_state.write_output = write_output

        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.WRITING,
                state="completed",
                data=f"撰写完成，report_len={len(write_output.generated_text)}",
            )
        )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Writing failed: {str(e)}"
        current_state.error.write_node_error = err_msg
        await state_queue.put(
            BackToFrontData(step=ExecutionState.WRITING, state="error", data=err_msg)
        )
        return {"value": current_state}
