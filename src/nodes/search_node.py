import json
import re
import textwrap
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import arxiv

from src.core.model_client import create_model_client
from src.core.openai_client import SystemMessage, UserMessage
from src.core.prompts import search_agent_prompt
from src.core.state_models import (
    BackToFrontData,
    ExecutionState,
    SearchInput,
    SearchOutput,
    SearchResult,
    SearchScope,
    State,
)
from src.nodes.base_node import BaseNode
from src.tasks.paper_search import PaperSearcher
from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)

# 允许的 arXiv 字段前缀白名单（小写，用于 query 验证）
_ALLOWED_FIELD_PREFIXES = {
    "all:",
    "ti:",
    "abs:",
    "au:",
    "submitteddate:",
    "submitted_date:",
}


class _LiteAgent:
    """极简 Agent 适配：单轮 prompt -> 文本。

    替代 autogen 的 AssistantAgent（去重后的轻量实现）。
    run(task) 返回兼容 ``response.messages[-1].content`` 访问形式的结果。
    """

    def __init__(self, model_client, system_message: str):
        self._client = model_client
        self._system_message = system_message

    async def run(self, task: str):
        messages = [
            SystemMessage(content=self._system_message),
            UserMessage(content=task, source="user"),
        ]
        result = await self._client.create(messages=messages)
        return SimpleNamespace(
            messages=[SimpleNamespace(content=result.content)]
        )


class SearchNode(BaseNode[SearchInput, SearchOutput]):
    """搜索节点：根据查询条件检索相关文档。"""

    name = "search_node"
    input_model = SearchInput
    output_model = SearchOutput

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.use_llm = self.config.get("use_llm", True)
        self.download_pdf = self.config.get("download_pdf", False)
        self.pdf_download_dir = self.config.get("pdf_download_dir", "data/papers")
        self.candidate_pool_min = max(1, int(self.config.get("candidate_pool_min", 20)))
        self.candidate_pool_multiplier = max(
            1, int(self.config.get("candidate_pool_multiplier", 5))
        )
        self.candidate_pool_max = max(
            self.candidate_pool_min,
            int(self.config.get("candidate_pool_max", 50)),
        )
        self.filter_batch_size = max(1, int(self.config.get("filter_batch_size", 20)))
        self.paper_searcher = PaperSearcher()
        self._search_agent: Optional[_LiteAgent] = None
        self._search_model_client = None

    def _get_search_agent(self) -> Optional[_LiteAgent]:
        """为当前 SearchNode 请求懒加载独立的查询 Agent。"""
        if self._search_agent is not None:
            return self._search_agent
        try:
            self._search_model_client = create_model_client()
            self._search_agent = _LiteAgent(
                model_client=self._search_model_client,
                system_message=search_agent_prompt,
            )
            return self._search_agent
        except Exception as e:
            self.logger.warning(f"创建搜索模型客户端失败: {e}")
            self._search_agent = None
            self._search_model_client = None
            return None

    async def close(self) -> None:
        """关闭当前请求创建的模型客户端。"""
        if self._search_model_client is not None:
            await self._search_model_client.close()
            self._search_model_client = None
            self._search_agent = None

    def _candidate_count(self, target_count: int) -> int:
        """根据最终目标数量计算 arXiv 候选池大小。"""
        return min(
            self.candidate_pool_max,
            max(
                self.candidate_pool_min,
                max(1, target_count) * self.candidate_pool_multiplier,
            ),
        )

    def _sort_criterion(self, user_request: str) -> arxiv.SortCriterion:
        """选择 arXiv 排序策略；仅明确要求最新时按提交时间排序。"""
        wants_latest = bool(
            re.search(r"最新|latest|newest", user_request or "", re.IGNORECASE)
        )
        configured = str(self.config.get("default_sort_by", "relevance")).lower()
        if wants_latest or configured == "submitted_date":
            return arxiv.SortCriterion.SubmittedDate
        return arxiv.SortCriterion.Relevance

    @staticmethod
    def _validate_query(query: str) -> bool:
        """验证 LLM 生成的 arXiv 查询表达式是否合法。

        检查项：
        - 括号平衡
        - 字段前缀在白名单内
        - 无可疑注入字符
        """
        if not query or not query.strip():
            return False

        # 括号平衡检查
        depth = 0
        for ch in query:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if depth < 0:
                return False
        if depth != 0:
            return False

        # 提取并验证字段前缀
        prefixes = re.findall(r"(?<![a-zA-Z])[a-zA-Z]+:", query)
        for p in prefixes:
            if p.lower() not in _ALLOWED_FIELD_PREFIXES:
                logger.warning(f"查询表达式中包含不允许的字段前缀: {p}")
                return False

        # 禁止控制字符（arXiv API 会拒绝）
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", query):
            return False

        return True

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """把查询中的引号短语转换为 arXiv API 支持的词项 AND 形式。

        arXiv API 不支持引号短语搜索（带引号的查询永远返回 0 篇），
        且字段前缀只作用于紧跟其后的第一个词。因此把
        ``all:"a b c"`` 规范化为 ``all:a AND all:b AND all:c``，
        并删除任何残留的双引号。

        Args:
            query: 原始 arXiv 查询表达式（可能含引号短语）。

        Returns:
            去掉引号、多词短语拆成词项 AND 的查询表达式。
        """
        if not query:
            return query

        def _expand(match: "re.Match[str]") -> str:
            field, phrase = match.group(1), match.group(2)
            words = [w for w in re.split(r"\s+", phrase.strip()) if w]
            if not words:
                return f"{field}:"
            return " AND ".join(f"{field}:{w}" for w in words)

        # 处理 field:"phrase" 形式（field 为字母前缀，如 all:/ti:/abs:/au:）
        sanitized = re.sub(r'([a-zA-Z]+):"([^"]*)"', _expand, query)
        # 删除任何残留的双引号（裸引号或未配对引号）
        return sanitized.replace('"', "")

    async def process(self, input_data: SearchInput) -> SearchOutput:
        """执行搜索：LLM 生成完整查询表达式 -> 调用 arxiv 搜索 -> 返回结构化结果。

        LLM 生成的查询会经过验证，无效则带反馈重试（最多 query_retry_limit 次）。
        若 LLM 多次失败后仍无法生成有效查询，直接报错，不做降级。
        """
        self.logger.info(f"[{self.name}] user_request={input_data.user_request}")

        query_keywords = list(input_data.query_keywords or [])
        start_date = input_data.search_scope.start_date
        end_date = input_data.search_scope.end_date
        target_count = max(1, input_data.search_scope.max_results)
        candidate_count = self._candidate_count(target_count)
        max_retries = self.config.get("query_retry_limit", 3)

        # ── 阶段 1：用 LLM 生成完整查询表达式（带验证 + 重试） ──
        arxiv_query: Optional[str] = None
        if input_data.user_request and self.use_llm:
            agent = self._get_search_agent()
            if agent is None:
                raise ValueError(
                    "LLM 搜索 Agent 初始化失败，无法生成查询表达式。"
                    "请检查统一 model 配置或设置 use_llm=False 使用关键词搜索。"
                )
            last_error: Optional[str] = None
            for attempt in range(1, max_retries + 1):
                try:
                    generated = await self._generate_search_query(
                        input_data.user_request, feedback=last_error
                    )
                    if generated and self._validate_query(generated):
                        arxiv_query = generated
                        self.logger.info(
                            f"[{self.name}] LLM 生成查询 (第{attempt}次): {arxiv_query}"
                        )
                        break
                    elif generated:
                        last_error = f"生成的查询表达式不合法: {generated}"
                        self.logger.warning(
                            f"[{self.name}] {last_error}，"
                            f"重试 ({attempt}/{max_retries})"
                        )
                    else:
                        last_error = "LLM 返回了空查询"
                        self.logger.warning(
                            f"[{self.name}] {last_error}，"
                            f"重试 ({attempt}/{max_retries})"
                        )
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(
                        f"[{self.name}] LLM 查询生成异常: {e}，"
                        f"重试 ({attempt}/{max_retries})"
                    )

            if arxiv_query is None:
                raise ValueError(
                    "LLM 无法生成有效的查询表达式"
                    f"（重试 {max_retries} 次后放弃）: {last_error}"
                )

        # ── 阶段 2：LLM 不可用时（use_llm=False 或 agent 初始化失败） ──
        else:
            if not query_keywords:
                raise ValueError("没有可用的搜索关键词")
            # arXiv API 不支持引号短语，关键词直接拼为词项（含空格时由
            # _sanitize_query 统一拆成词项 AND）
            arxiv_query = "(" + " AND ".join(f"all:{q}" for q in query_keywords) + ")"
            self.logger.info(f"[{self.name}] 关键词拼装查询: {arxiv_query}")

        # ── 阶段 3：排序策略 ──
        sort_by = self._sort_criterion(input_data.user_request or "")
        sort_order = arxiv.SortOrder.Descending

        # ── 阶段 3.5：消毒查询（LLM 可能仍生成引号短语，统一转为词项 AND） ──
        arxiv_query = self._sanitize_query(arxiv_query)
        self.logger.info(f"[{self.name}] 最终查询: {arxiv_query}")

        # ── 阶段 4：执行搜索 ──
        papers = await self.paper_searcher.search_papers(
            query=arxiv_query,
            max_results=candidate_count,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        # ── 阶段 5：相关性过滤（纵深防御） ──
        papers, discarded = await self._filter_relevant_papers(
            papers, input_data.user_request or "", query_keywords
        )
        filtered_count = len(papers)
        papers = papers[:target_count]

        # ── 阶段 6：转换结果并下载 PDF ──
        results = []
        for paper in papers:
            result = self._to_search_result(paper)
            if self.download_pdf and result.pdf_url:
                result.pdf_path = await self.paper_searcher.download_pdf(
                    pdf_url=result.pdf_url,
                    paper_id=result.paper_id,
                    download_dir=self.pdf_download_dir,
                )
            results.append(result)

        return SearchOutput(
            results=results,
            total_count=len(results),
            status="completed",
            metadata={
                "arxiv_query": arxiv_query,
                "querys": query_keywords,
                "start_date": start_date,
                "end_date": end_date,
                "candidate_count": candidate_count,
                "filtered_kept": filtered_count,
                "returned_count": len(papers),
                "filtered_discarded": discarded,
            },
        )

    async def _generate_search_query(
        self, user_request: str, feedback: Optional[str] = None
    ) -> Optional[str]:
        """使用 LLM Agent 根据用户请求生成完整 arXiv 查询表达式。

        Args:
            user_request: 用户的原始需求。
            feedback: 前一次生成失败的原因，用于引导 LLM 自我修正。
        """
        agent = self._get_search_agent()
        if agent is None:
            return None
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = textwrap.dedent(
                f"""\
                当前日期：{current_date}

                请根据用户查询需求，生成 arXiv 检索用的 query 表达式。
                用户查询需求：{user_request}
                """
            )
            if feedback:
                prompt += (
                    f"\n前一次生成的查询无效，原因：{feedback}\n"
                    "请修正后重新生成，只输出 query 本身。"
                )
            response = await agent.run(task=prompt)
            content = response.messages[-1].content
            if isinstance(content, str):
                return content.strip()
            return None
        except Exception as e:
            self.logger.warning(f"LLM 查询生成失败: {e}")
            return None

    def _to_search_result(self, paper: Dict[str, Any]) -> SearchResult:
        """将 PaperSearcher 返回的论文字典转换为 SearchResult。"""
        return SearchResult(
            paper_id=str(paper.get("paper_id", "")),
            title=str(paper.get("title", "")),
            authors=list(paper.get("authors", [])),
            summary=str(paper.get("summary", "")),
            url=str(paper.get("url", "")),
            pdf_url=paper.get("pdf_url"),
            pdf_path=None,
            published=paper.get("published_date") or str(paper.get("published") or ""),
            metadata={
                "primary_category": paper.get("primary_category"),
                "categories": paper.get("categories"),
                "doi": paper.get("doi"),
                "published_year": paper.get("published"),
            },
        )

    async def _filter_relevant_papers(
        self,
        papers: List[Dict[str, Any]],
        user_request: str,
        query_keywords: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """使用 LLM 批量过滤论文，只保留与用户请求严格相关的论文。

        宁可少保留，也不混入无关论文。如果 LLM 调用失败则保留全部。

        Returns:
            (kept, discarded) 二元组。kept 为保留论文列表，discarded 为丢弃论文摘要列表
            （每项含 title/paper_id/index）。LLM 异常时 discarded 为空列表。
        """
        if not papers or not user_request:
            return papers, []

        filtered_all: List[Dict[str, Any]] = []
        discarded_all: List[Dict[str, Any]] = []
        for start in range(0, len(papers), self.filter_batch_size):
            batch = papers[start : start + self.filter_batch_size]
            filtered, discarded = await self._filter_relevant_batch(
                batch, user_request, query_keywords
            )
            filtered_all.extend(filtered)
            for record in discarded:
                adjusted = {**record, "index": int(record["index"]) + start}
                discarded_all.append(adjusted)

        self.logger.info(
            f"[{self.name}] 相关性过滤汇总：{len(papers)} → {len(filtered_all)} 篇"
        )
        return filtered_all, discarded_all

    async def _filter_relevant_batch(
        self,
        papers: List[Dict[str, Any]],
        user_request: str,
        query_keywords: List[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """对一批候选论文执行相关性过滤。"""

        # 构建紧凑的论文摘要列表，便于 LLM 批量判断
        lines = []
        for i, p in enumerate(papers):
            title = (p.get("title") or "").strip()
            summary = (p.get("summary") or "").strip()[:500]
            lines.append(f"[{i}] {title}\n    {summary[:200]}")

        prompt = textwrap.dedent(f"""\
            用户需求：{user_request}

            以下是 arXiv 搜索结果中的论文标题和摘要：

            {chr(10).join(lines)}

            请判断每篇论文是否**同时涉及**用户需求中的所有核心概念。
            只保留严格相关的论文——宁可漏掉边界论文，也不要混入无关的。

            请只输出一个 JSON 数组，包含严格相关的论文索引，
            例如 [0, 3]。如果都不相关输出 []。
            """)

        model_client = None
        try:
            model_client = create_model_client()
            filter_agent = _LiteAgent(
                model_client=model_client,
                system_message="你是一个论文相关性判断助手。你的任务是根据用户需求判断论文是否严格相关。",
            )
            response = await filter_agent.run(task=prompt)
            content = response.messages[-1].content
            if isinstance(content, str):
                match = re.search(r"\[[\d,\s]*\]", content)
                if match:
                    indices = json.loads(match.group())
                    filtered = [papers[i] for i in indices if 0 <= i < len(papers)]
                    # 计算被丢弃的论文并记录到 debug 日志
                    kept_set = set(indices)
                    discard_records = []
                    for i, p in enumerate(papers):
                        if i not in kept_set:
                            record = {
                                "title": p.get("title", ""),
                                "paper_id": p.get("paper_id", ""),
                                "index": i,
                            }
                            discard_records.append(record)
                            title = p.get("title", "")
                            paper_id = p.get("paper_id", "")
                            self.logger.debug(
                                f"[{self.name}] 过滤丢弃 [{i}]: {title} ({paper_id})"
                            )
                    self.logger.info(
                        f"[{self.name}] 相关性过滤：{len(papers)} → {len(filtered)} 篇"
                    )
                    return filtered, discard_records
            self.logger.info(f"[{self.name}] 相关性过滤：未能解析 LLM 输出，保留全部")
            return papers, []
        except Exception as e:
            self.logger.warning(f"[{self.name}] 相关性过滤异常，保留全部: {e}")
            return papers, []
        finally:
            if model_client is not None:
                await model_client.close()


async def search_node(state: State) -> State:
    """LangGraph 适配函数：搜索节点。

    命令行阶段直接打印状态；前端接入时设置
    config={"search_node": {"frontend_enabled": True}}
    即可切换为向 state_queue 推送消息。
    """
    state_queue = state["state_queue"]
    current_state = state["value"]

    node_config = current_state.config.get("search_node", {})
    frontend_enabled = node_config.get("frontend_enabled", False)

    async def _report_status(status: str, data: Any = None) -> None:
        """状态上报：frontend_enabled=True 时推送给前端，否则打印到命令行。"""
        if frontend_enabled:
            await state_queue.put(
                BackToFrontData(step=ExecutionState.SEARCHING, state=status, data=data)
            )
        else:
            print(f"[search_node] {status}: {data}")

    try:
        current_state.current_step = ExecutionState.SEARCHING
        await _report_status("initializing", None)

        node = SearchNode(node_config)

        search_input = SearchInput(
            query_keywords=[current_state.user_request],
            search_scope=SearchScope(
                sources=node_config.get("sources", ["arxiv"]),
                max_results=current_state.max_papers,
            ),
            user_request=current_state.user_request,
        )

        try:
            search_output = await node.run(search_input)
        finally:
            await node.close()
        current_state.search_output = search_output

        if search_output.total_count == 0:
            err_msg = "没有找到相关论文,请尝试其他查询条件"
            current_state.error.search_node_error = err_msg
            await _report_status("error", err_msg)
        else:
            await _report_status(
                "completed", f"搜索完成，共找到 {search_output.total_count} 条结果"
            )
        return {"value": current_state}

    except Exception as e:
        err_msg = f"Search failed: {e}"
        current_state.error.search_node_error = err_msg
        await _report_status("error", err_msg)
        return {"value": current_state}
