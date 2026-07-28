import asyncio
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz
from autogen_core import Image as AutoGenImage
from autogen_core.models import SystemMessage, UserMessage
from PIL import Image as PILImage
from pydantic import ValidationError

from src.core.model_client import create_model_client
from src.core.prompts import read_agent_prompt, verify_prompt
from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    KeyInformation,
    ReadingStrategy,
    ReadInput,
    ReadOutput,
    State,
    VerificationItem,
    VerifyResult,
)
from src.nodes.base_node import BaseNode


class _DetailImage(AutoGenImage):
    """覆写 to_openai_format 使 detail 参数生效。"""

    def __init__(self, pil_image: PILImage.Image, detail: str = "low"):
        super().__init__(pil_image)
        self._detail = detail

    def to_openai_format(self, detail: str = "auto"):
        return super().to_openai_format(self._detail)


class ReadNode(BaseNode[ReadInput, ReadOutput]):
    """阅读节点：从 PDF 提取文本，用 LLM 逐篇抽取 KeyInformation。"""

    name = "read_node"
    input_model = ReadInput
    output_model = ReadOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = self.config or {}
        self.concurrency: int = int(cfg.get("concurrency", 5))
        self.max_tokens_threshold: int = int(cfg.get("max_tokens_threshold", 25000))

        image_cfg = cfg.get("image", {}) or {}
        self.use_images: bool = bool(cfg.get("use_images", True))
        self.image_dpi: int = int(image_cfg.get("dpi", 150))
        self.image_detail: str = image_cfg.get("detail", "low")
        self.image_max_size: int = int(image_cfg.get("max_size", 1024))

        self._model_client = None
        self._markdown_cache: Dict[str, str] = {}
        self._image_cache: Dict[str, List[_DetailImage]] = {}

    # ── 核心入口 ──────────────────────────────────────────────

    async def process(self, input_data: ReadInput) -> ReadOutput:
        documents = input_data.documents
        strategy = input_data.reading_strategy

        if not documents:
            self.logger.info("[read_node] 无文档，跳过处理")
            return ReadOutput(key_info=[], status="completed", failed_paper_ids=[])

        self.logger.info(
            f"[read_node] 开始处理 {len(documents)} 篇论文，并发={self.concurrency}"
        )

        sem = asyncio.Semaphore(self.concurrency)

        async def _process_with_limit(paper):
            async with sem:
                return await self._process_single(paper, strategy)

        results = await asyncio.gather(
            *[_process_with_limit(p) for p in documents],
            return_exceptions=True,
        )

        key_info: List[KeyInformation] = []
        failed_paper_ids: List[str] = []

        for paper, result in zip(documents, results):
            if isinstance(result, Exception):
                self.logger.error(
                    f"[read_node] 论文 {paper.paper_id} 未处理异常: {result}"
                )
                failed_paper_ids.append(paper.paper_id)
            elif result is None:
                failed_paper_ids.append(paper.paper_id)
            else:
                key_info.append(result)

        self.logger.info(
            f"[read_node] 完成 — 成功={len(key_info)} 失败={len(failed_paper_ids)}"
        )
        return ReadOutput(
            key_info=key_info,
            status="completed",
            failed_paper_ids=failed_paper_ids,
        )

    # ── 单篇处理管线 ──────────────────────────────────────────

    async def _process_single(
        self, paper, strategy: ReadingStrategy
    ) -> Optional[KeyInformation]:
        """处理单篇论文，失败返回 None。"""
        paper_id = getattr(paper, "paper_id", "")
        pdf_path = getattr(paper, "pdf_path", None)

        if not pdf_path:
            self.logger.warning(f"[read_node] {paper_id}: pdf_path 为空")
            return None

        try:
            full_text, page_texts, page_images = await asyncio.to_thread(
                self._extract_pdf_content, pdf_path
            )
            self._markdown_cache[paper_id] = full_text
            if self.use_images:
                self._image_cache[paper_id] = page_images

            if self.use_images and page_texts:
                page_groups = self._group_pages_into_chunks(page_texts, page_images)
                chunk_results = []
                for group_texts, group_images in page_groups:
                    group_full = "\n\n".join(group_texts)
                    result = await self._call_llm_extract(
                        group_full,
                        strategy,
                        paper,
                        page_texts=group_texts,
                        page_images=group_images,
                    )
                    if result:
                        chunk_results.append(result)
                if not chunk_results:
                    return None
                return self._merge_chunk_results(chunk_results)
            else:
                if self._estimate_tokens(full_text) > self.max_tokens_threshold:
                    self.logger.info(
                        f"[read_node] {paper_id}: 论文过长，按 section 切块"
                    )
                    chunks = self._chunk_text_for_model(full_text)
                    chunk_results = []
                    for chunk in chunks:
                        result = await self._call_llm_extract(chunk, strategy, paper)
                        if result:
                            chunk_results.append(result)
                    if not chunk_results:
                        return None
                    return self._merge_chunk_results(chunk_results)
                else:
                    return await self._call_llm_extract(full_text, strategy, paper)

        except Exception as e:
            self.logger.error(f"[read_node] {paper_id}: 处理异常 — {e}")
            return None

    # ── PDF 提取 ──────────────────────────────────────────────

    def _resize_pil_image(self, pil_img: PILImage.Image) -> PILImage.Image:
        """缩放图片到 self.image_max_size 以内，控制内存。"""
        w, h = pil_img.size
        max_dim = max(w, h)
        if max_dim <= self.image_max_size:
            return pil_img
        ratio = self.image_max_size / max_dim
        return pil_img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)

    def _render_pdf_pages(self, pdf_path: str) -> List[_DetailImage]:
        """将 PDF 每页渲染为图片，返回 _DetailImage 列表。"""
        doc = fitz.open(pdf_path)
        images: List[_DetailImage] = []
        for page in doc:
            pix = page.get_pixmap(dpi=self.image_dpi)
            pil_img = PILImage.open(io.BytesIO(pix.tobytes("png")))
            pil_img = self._resize_pil_image(pil_img)
            images.append(_DetailImage(pil_img, detail=self.image_detail))
        doc.close()
        return images

    def _extract_pdf_content(
        self, pdf_path: str
    ) -> Tuple[str, List[str], List[_DetailImage]]:
        """
        提取 PDF 并返回:
            full_text:   全文 markdown
            page_texts:  每页 markdown 文本列表
            page_images: 每页渲染图片列表
        """
        if self.use_images:
            # pymupdf4llm 会加载 ONNX 等较重依赖。仅多模态模式需要其
            # Markdown 布局分析，避免纯文本部署承担数百 MB 的内存峰值。
            import pymupdf4llm

            page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            page_texts = [chunk["text"] for chunk in page_chunks]
            page_images = self._render_pdf_pages(pdf_path)
            if len(page_texts) != len(page_images):
                self.logger.warning(
                    f"page_texts({len(page_texts)}) != "
                    f"page_images({len(page_images)}), truncating"
                )
                min_count = min(len(page_texts), len(page_images))
                page_texts = page_texts[:min_count]
                page_images = page_images[:min_count]
        else:
            # Render 小内存实例使用轻量文本提取。逐页处理避免
            # pymupdf4llm 的版面分析模型在长论文上触发 OOM。
            with fitz.open(pdf_path) as doc:
                page_texts = [page.get_text("text", sort=True) for page in doc]
            page_images = []

        full_text = "\n\n".join(page_texts)
        return full_text, page_texts, page_images

    # ── Token 估算 ────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数：字符数 / 4。"""
        return len(text) // 4

    # ── Section 切块 ──────────────────────────────────────────

    def _split_by_section(self, md_text: str) -> List[str]:
        """按 ## 二级标题切块；超长块按 ### 三级标题再切。"""
        lines = md_text.split("\n")
        chunks: List[str] = []
        current_lines: List[str] = []

        for line in lines:
            if line.startswith("## ") and current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            chunks.append("\n".join(current_lines))

        # 如果第一个 chunk 只是论文大标题（# 开头且无实质内容），合并到下一个 chunk
        if len(chunks) > 1 and chunks[0].startswith("# ") and "## " not in chunks[0]:
            title = chunks.pop(0)
            chunks[0] = title + "\n" + chunks[0]

        # 超长块按 ### 再切
        result: List[str] = []
        for chunk in chunks:
            if self._estimate_tokens(chunk) > self.max_tokens_threshold:
                result.extend(self._split_by_level3(chunk))
            else:
                result.append(chunk)
        return result

    def _split_by_level3(self, text: str) -> List[str]:
        """按 ### 三级标题进一步切分。"""
        lines = text.split("\n")
        sub_chunks: List[str] = []
        current: List[str] = []

        for line in lines:
            if line.startswith("### ") and current:
                sub_chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            sub_chunks.append("\n".join(current))
        return sub_chunks if sub_chunks else [text]

    def _chunk_text_for_model(self, md_text: str) -> List[str]:
        """按章节切分并重新打包，确保每个模型请求不超过 token 阈值。"""
        sections = self._split_by_section(md_text)
        pieces: List[str] = []
        max_chars = max(4, self.max_tokens_threshold * 4)

        for section in sections:
            if self._estimate_tokens(section) <= self.max_tokens_threshold:
                pieces.append(section)
                continue
            for start in range(0, len(section), max_chars):
                pieces.append(section[start : start + max_chars])

        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for piece in pieces:
            piece_tokens = max(1, self._estimate_tokens(piece))
            if current and current_tokens + piece_tokens > self.max_tokens_threshold:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            current.append(piece)
            current_tokens += piece_tokens

        if current:
            chunks.append("\n\n".join(current))
        return chunks or [md_text]

    @staticmethod
    def _format_authors(authors: List[str], limit: int = 20) -> str:
        """限制超长作者列表，避免其重复占用模型上下文。"""
        if not authors:
            return "未知"
        visible = ", ".join(authors[:limit])
        remaining = len(authors) - limit
        return f"{visible}（另有 {remaining} 位作者）" if remaining > 0 else visible

    # ── LLM 提取 ──────────────────────────────────────────────

    async def _call_llm_extract(
        self,
        text: str,
        strategy: ReadingStrategy,
        paper,
        page_texts: Optional[List[str]] = None,
        page_images: Optional[List[_DetailImage]] = None,
    ) -> Optional[KeyInformation]:
        """调用 LLM 抽取 KeyInformation，最多重试 1 次。"""
        paper_id = getattr(paper, "paper_id", "")
        client = self._get_model_client()

        if self.use_images and page_texts and page_images:
            content = self._build_multimodal_content(
                strategy, paper, page_texts, page_images
            )
        else:
            content = self._build_user_prompt(text, strategy, paper)

        messages = [
            SystemMessage(content=read_agent_prompt),
            UserMessage(content=content, source="user"),
        ]

        for attempt in range(2):
            try:
                result = await client.create(messages=messages, json_output=True)
                content_str = result.content

                # 清理可能的 markdown 代码块包装
                content_str = self._strip_json_fence(content_str)
                parsed = json.loads(content_str)
                parsed = self._normalize_types(parsed)
                parsed["paper_id"] = paper_id
                key_info = KeyInformation(**parsed)
                self.logger.info(f"[read_node] {paper_id}: 提取成功")
                return key_info

            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.warning(
                    f"[read_node] {paper_id}: 第 {attempt + 1} 次解析失败 — {e}"
                )
                if attempt == 0:
                    continue
                self.logger.error(f"[read_node] {paper_id}: 重试后仍失败")
                return None
            except Exception as e:
                self.logger.error(f"[read_node] {paper_id}: LLM 调用异常 — {e}")
                return None

        return None

    def _build_header_prompt(self, strategy: ReadingStrategy, paper) -> str:
        """构造多模态路径的头部 prompt（不含论文全文）。"""
        title = getattr(paper, "title", "")
        authors = getattr(paper, "authors", [])
        summary = getattr(paper, "summary", "")
        authors_str = self._format_authors(authors)
        focus_areas = (
            ", ".join(strategy.focus_areas)
            if strategy.focus_areas
            else "method, result, limitation"
        )

        depth = getattr(strategy, "depth", "medium")
        depth_instruction = {
            "shallow": "做简要提取，每个字段 2-3 句话即可。",
            "medium": (
                "做中等详细度的提取，保留关键技术细节和主要实验数据，"
                "每个字段 5-10 句话。"
            ),
            "deep": (
                "做深度提取，把这篇论文当作唯一的信息来源。每个字段必须详尽：\n"
                "- core_problem: 说清楚问题的背景、现有方法的不足、本文的目标\n"
                "- key_methodology: 完整描述技术架构（所有模块）、训练策略、"
                "使用的数据集、超参数、评估协议\n"
                "- main_results: 逐实验、逐数据集、逐指标列出所有数值，"
                "不要省略任何一行结果表格的数据\n"
                "- limitations: 逐一列出每个局限，说明为什么存在、影响是什么\n"
                "- contributions: 列出所有贡献点"
            ),
        }.get(
            depth,
            "做中等详细度的提取，保留关键技术细节和主要实验数据，每个字段 5-10 句话。",
        )

        return (
            f"## 论文元信息\n"
            f"- 标题：{title}\n"
            f"- 作者：{authors_str}\n"
            f"- 摘要：{summary}\n\n"
            f"## 提取要求\n"
            f"- 详细程度：{depth_instruction}\n"
            f"- 重点关注维度：{focus_areas}\n\n"
            f"## 论文全文（逐页展示，每页文本后附该页原文截图）\n"
            f"请逐页阅读文本和截图。重点关注截图中的图表、表格和流程图，"
            f"提取其中的关键数值、趋势和结构信息，"
            f"尤其关注文本中未完整呈现的实验结果和架构设计。\n"
            f"注意：截图受渲染精度影响可能不清晰，请以文本中的数值为准，"
            f"截图仅作为补充参考。"
        )

    def _build_multimodal_content(
        self,
        strategy: ReadingStrategy,
        paper,
        page_texts: List[str],
        page_images: List[_DetailImage],
    ) -> List[Union[str, _DetailImage]]:
        """构建多模态消息内容：header + 逐页图文交错。"""
        header = self._build_header_prompt(strategy, paper)
        content: List[Union[str, _DetailImage]] = [header]
        for i, (text, img) in enumerate(zip(page_texts, page_images)):
            content.append(f"\n--- 第 {i + 1} 页 ---\n{text}")
            content.append(img)
        return content

    def _group_pages_into_chunks(
        self,
        page_texts: List[str],
        page_images: List[_DetailImage],
    ) -> List[Tuple[List[str], List[_DetailImage]]]:
        """按 token 预算将页面分组，每组附带对应图片。"""
        IMAGE_TOKEN_BUDGET = 85
        chunks: List[Tuple[List[str], List[_DetailImage]]] = []
        cur_texts: List[str] = []
        cur_imgs: List[_DetailImage] = []
        cur_tokens = 0

        for text, img in zip(page_texts, page_images):
            entry_tokens = self._estimate_tokens(text) + IMAGE_TOKEN_BUDGET
            if cur_tokens + entry_tokens > self.max_tokens_threshold and cur_texts:
                chunks.append((cur_texts, cur_imgs))
                cur_texts, cur_imgs, cur_tokens = [text], [img], entry_tokens
            else:
                cur_texts.append(text)
                cur_imgs.append(img)
                cur_tokens += entry_tokens

        if cur_texts:
            chunks.append((cur_texts, cur_imgs))
        return chunks

    def _build_user_prompt(self, text: str, strategy: ReadingStrategy, paper) -> str:
        """构造 LLM 用户消息。"""
        title = getattr(paper, "title", "")
        authors = getattr(paper, "authors", [])
        summary = getattr(paper, "summary", "")
        authors_str = self._format_authors(authors)
        focus_areas = (
            ", ".join(strategy.focus_areas)
            if strategy.focus_areas
            else "method, result, limitation"
        )

        depth = getattr(strategy, "depth", "medium")
        depth_instruction = {
            "shallow": "做简要提取，每个字段 2-3 句话即可。",
            "medium": (
                "做中等详细度的提取，保留关键技术细节和主要实验数据，"
                "每个字段 5-10 句话。"
            ),
            "deep": "做深度提取，把这篇论文当作唯一的信息来源。每个字段必须详尽：\n"
            "- core_problem: 说清楚问题的背景、现有方法的不足、本文的目标\n"
            "- key_methodology: 完整描述技术架构（所有模块）、训练策略、"
            "使用的数据集、超参数、评估协议\n"
            "- main_results: 逐实验、逐数据集、逐指标列出所有数值，"
            "不要省略任何一行结果表格的数据\n"
            "- limitations: 逐一列出每个局限，说明为什么存在、影响是什么\n"
            "- contributions: 列出所有贡献点",
        }.get(
            depth,
            "做中等详细度的提取，保留关键技术细节和主要实验数据，每个字段 5-10 句话。",
        )

        return f"""## 论文元信息
- 标题：{title}
- 作者：{authors_str}
- 摘要：{summary}

## 论文全文（Markdown）
{text}

## 提取要求
- 详细程度：{depth_instruction}
- 重点关注维度：{focus_areas}"""

    @staticmethod
    def _strip_json_fence(text: str) -> str:
        """去除 ```json ... ``` 包裹（DeepSeek 偶尔会加）。"""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        return text

    @staticmethod
    def _normalize_types(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """容错：LLM 可能把 string 字段输出为 list，或反之。"""
        # string 字段：如果 LLM 返回了 list，用分号拼接
        for field in ("core_problem", "key_methodology", "main_results", "limitations"):
            if field in parsed and isinstance(parsed[field], list):
                parsed[field] = "；".join(parsed[field])
        # contributions 必须是 list，如果 LLM 返回了 string，拆开
        if "contributions" in parsed and isinstance(parsed["contributions"], str):
            parts = [
                p.strip()
                for p in re.split(r"[；;]", parsed["contributions"])
                if p.strip()
            ]
            parsed["contributions"] = parts if parts else [parsed["contributions"]]
        return parsed

    # ── 切块合并 ──────────────────────────────────────────────

    def _merge_chunk_results(
        self, chunk_results: List[KeyInformation]
    ) -> KeyInformation:
        """合并同一论文多个 section 的提取结果。"""
        if not chunk_results:
            return KeyInformation()
        if len(chunk_results) == 1:
            return chunk_results[0]

        paper_id = chunk_results[0].paper_id

        # core_problem: 优先 Abstract / Introduction
        core_problem = self._first_non_empty(
            chunk_results,
            "core_problem",
            preferred_sections=["abstract", "introduction"],
        )

        # key_methodology: 拼接所有非空
        key_methodology = self._concat_non_empty(chunk_results, "key_methodology")

        # main_results: 拼接所有非空
        main_results = self._concat_non_empty(chunk_results, "main_results")

        # limitations: 优先 Discussion / Conclusion
        limitations = self._first_non_empty(
            chunk_results,
            "limitations",
            preferred_sections=["discussion", "conclusion"],
        )

        # contributions: 汇总去重
        all_contributions: List[str] = []
        for c in chunk_results:
            all_contributions.extend(c.contributions)
        contributions = list(dict.fromkeys(all_contributions))

        # evidence_sections: 合并所有
        merged_evidence: Dict[str, str] = {}
        for c in chunk_results:
            for k, v in c.evidence_sections.items():
                if v and k not in merged_evidence:
                    merged_evidence[k] = v

        return KeyInformation(
            paper_id=paper_id,
            core_problem=core_problem,
            key_methodology=key_methodology,
            main_results=main_results,
            limitations=limitations,
            contributions=contributions,
            evidence_sections=merged_evidence,
        )

    @staticmethod
    def _first_non_empty(
        results: List[KeyInformation],
        field: str,
        preferred_sections: Optional[List[str]] = None,
    ) -> str:
        """从 results 中取 field 的第一个有效值；可优先匹配 sections。"""
        if preferred_sections:
            for section in preferred_sections:
                for r in results:
                    evidence = r.evidence_sections.get(field, "").lower()
                    if section in evidence:
                        val = getattr(r, field, "")
                        if val and val != "未提及":
                            return val
        # 回退：任意第一个非空
        for r in results:
            val = getattr(r, field, "")
            if val and val != "未提及":
                return val
        return ""

    @staticmethod
    def _concat_non_empty(results: List[KeyInformation], field: str) -> str:
        """拼接所有非空 field 值，用换行分隔。"""
        parts = []
        for r in results:
            val = getattr(r, field, "")
            if val and val != "未提及":
                parts.append(val)
        return "\n".join(parts)

    # ── 验证 ──────────────────────────────────────────────────

    async def _verify_one_paper(
        self, md_text: str, ki: KeyInformation
    ) -> Optional[VerifyResult]:
        """用 LLM 验证一篇论文的提取结果是否忠实于原文。"""
        paper_id = ki.paper_id

        claims_to_verify = {
            "key_methodology": ki.key_methodology,
            "main_results": ki.main_results,
            "limitations": ki.limitations,
        }

        text_part = (
            f"## 论文全文（Markdown）\n\n{md_text}\n\n"
            f"## 提取结果（待验证）\n\n"
            f"{json.dumps(claims_to_verify, ensure_ascii=False, indent=2)}"
        )

        page_images = self._image_cache.get(paper_id, [])
        if self.use_images and page_images:
            text_part += "\n\n## 附：论文原文页面截图（可辅助核对图表和数值）"
            user_content: Union[str, List] = [text_part] + page_images
        else:
            user_content = text_part

        messages = [
            SystemMessage(content=verify_prompt),
            UserMessage(content=user_content, source="user"),
        ]

        client = self._get_model_client()
        try:
            result = await client.create(messages=messages, json_output=True)
            content = self._strip_json_fence(result.content)
            parsed = json.loads(content)

            items = []
            for field in ("key_methodology", "main_results", "limitations"):
                field_result = parsed.get(field, {})
                items.append(
                    VerificationItem(
                        field=field,
                        verified=bool(field_result.get("verified", False)),
                        exact_quote=str(field_result.get("exact_quote", "")),
                        reason=str(field_result.get("reason", "")),
                        original_claim=str(claims_to_verify.get(field, "")),
                    )
                )

            from datetime import datetime

            verify_result = VerifyResult(
                paper_id=paper_id,
                items=items,
                passed=all(item.verified for item in items),
                retry_count=0,
                verified_at=datetime.now().isoformat(),
            )
            verification_status = "通过" if verify_result.passed else "不通过"
            self.logger.info(f"[read_node] {paper_id}: 验证{verification_status}")
            return verify_result

        except Exception as e:
            self.logger.error(f"[read_node] {paper_id}: 验证调用异常 — {e}")
            return None

    # ── 模型客户端 ────────────────────────────────────────────

    def _get_model_client(self):
        """懒加载 LLM 客户端。"""
        if self._model_client is None:
            self._model_client = create_model_client()
        return self._model_client


# ── LangGraph 适配函数 ────────────────────────────────────────


async def read_node(state: State) -> State:
    """LangGraph 适配函数：阅读节点，含提取 + 验证。"""
    state_queue = state["state_queue"]
    current_state = state["value"]

    try:
        current_state.current_step = ExecutionState.READING
        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.READING, state="initializing", data=None
            )
        )

        read_config = current_state.config.get("read_node", {})
        verify_cfg = read_config.get("verify", {})
        verify_enabled = verify_cfg.get("enabled", True)

        node = ReadNode(read_config)

        # ── 筛选尚未通过验证的论文 ──
        existing_results = current_state.verify_results or {}
        papers_to_process = [
            doc
            for doc in current_state.search_output.results
            if not (
                doc.paper_id in existing_results
                and existing_results[doc.paper_id].passed
            )
        ]

        if papers_to_process:
            focus_areas = read_config.get(
                "focus_areas", ["method", "result", "limitation"]
            )
            read_input = ReadInput(
                documents=papers_to_process,
                reading_strategy=ReadingStrategy(
                    focus_areas=focus_areas,
                    depth=read_config.get("depth", "deep"),
                ),
            )

            read_output = await node.run(read_input)

            # ── 验证 ──
            if verify_enabled:
                for ki in read_output.key_info:
                    md_text = node._markdown_cache.get(ki.paper_id, "")
                    if md_text:
                        verify_result = await node._verify_one_paper(md_text, ki)
                        if verify_result:
                            prev = existing_results.get(ki.paper_id)
                            if prev:
                                verify_result.retry_count = prev.retry_count + 1
                            current_state.verify_results[ki.paper_id] = verify_result
                    node._image_cache.pop(ki.paper_id, None)

            # ── 合并新旧结果 ──
            old_key_info = current_state.read_output.key_info or []
            updated_results = current_state.verify_results
            passed_ids = {pid for pid, vr in updated_results.items() if vr.passed}
            merged = [ki for ki in old_key_info if ki.paper_id in passed_ids]
            merged.extend(read_output.key_info)

            current_state.read_output = ReadOutput(
                key_info=merged,
                status="completed",
                failed_paper_ids=read_output.failed_paper_ids,
            )

        # ── 统计状态 ──
        total_papers = len(current_state.search_output.results)
        verified_count = sum(
            1 for vr in current_state.verify_results.values() if vr.passed
        )
        failed_verify = sum(
            1 for vr in current_state.verify_results.values() if not vr.passed
        )
        extract_failed = len(current_state.read_output.failed_paper_ids)

        await state_queue.put(
            BackToFrontData(
                step=ExecutionState.READING,
                state="completed",
                data=f"阅读完成 — 已验证 {verified_count}/{total_papers} 篇"
                f"（待重试 {failed_verify} 篇，提取失败 {extract_failed} 篇）",
            )
        )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Reading failed: {str(e)}"
        current_state.error.read_node_error = err_msg
        await state_queue.put(
            BackToFrontData(step=ExecutionState.READING, state="error", data=err_msg)
        )
        return {"value": current_state}
