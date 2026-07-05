import ast
import re
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

import arxiv
from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field

from src.core.model_client import create_search_model_client
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


class SearchQuery(BaseModel):
    """查询条件类，存储用户查询需求。"""

    querys: List[str] = Field(default_factory=list, description="查询条件列表")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")


_search_agent: Optional[AssistantAgent] = None


def get_search_agent() -> Optional[AssistantAgent]:
    """懒加载并返回搜索 Agent；配置缺失时返回 None。"""
    global _search_agent
    if _search_agent is not None:
        return _search_agent
    try:
        model_client = create_search_model_client()
        _search_agent = AssistantAgent(
            name="search_agent",
            model_client=model_client,
            system_message=search_agent_prompt,
        )
        return _search_agent
    except Exception as e:
        logger.warning(f"创建搜索模型客户端失败，将使用输入关键词直接搜索: {e}")
        _search_agent = None
        return None


def parse_search_query(s: str) -> SearchQuery:
    """将搜索模型传回的字符串转为 SearchQuery 对象。"""
    querys_match = re.search(r"querys\s*=\s*(\[[^\]]*\])", s)
    start_match = re.search(r"start_date\s*=\s*(?:'([^']*)'|None)", s)
    end_match = re.search(r"end_date\s*=\s*(?:'([^']*)'|None)", s)

    querys: List[str] = []
    if querys_match:
        try:
            querys = ast.literal_eval(querys_match.group(1))
        except Exception:
            querys = []

    def _normalize_date(value: Optional[str]) -> Optional[str]:
        """将空字符串统一转为 None。"""
        return value.strip() if value and value.strip() and value.strip().lower() != "none" else None

    start_date = _normalize_date(start_match.group(1)) if start_match else None
    end_date = _normalize_date(end_match.group(1)) if end_match else None

    return SearchQuery(querys=querys, start_date=start_date, end_date=end_date)


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
        self.paper_searcher = PaperSearcher()

    @staticmethod
    def _normalize_date(value: Optional[str]) -> Optional[str]:
        """将空字符串统一转为 None。"""
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped and stripped.lower() != "none" else None

    @staticmethod
    def _validate_date_order(start_date: Optional[str], end_date: Optional[str]) -> tuple:
        """校验并修正日期顺序。"""
        if start_date and end_date and start_date > end_date:
            logger.warning(f"开始日期 {start_date} 晚于结束日期 {end_date}，已自动交换")
            return end_date, start_date
        return start_date, end_date

    async def process(self, input_data: SearchInput) -> SearchOutput:
        """执行搜索：LLM 生成查询条件 -> 调用 arxiv 搜索 -> 返回结构化结果。"""
        self.logger.info(f"[{self.name}] query_keywords={input_data.query_keywords}")

        query_keywords = list(input_data.query_keywords or [])
        start_date = self._normalize_date(input_data.search_scope.start_date)
        end_date = self._normalize_date(input_data.search_scope.end_date)
        max_results = input_data.search_scope.max_results

        if input_data.user_request and self.use_llm:
            generated_query = await self._generate_search_queries(input_data.user_request)
            if generated_query and generated_query.querys:
                query_keywords = generated_query.querys
                start_date = self._normalize_date(generated_query.start_date) or start_date
                end_date = self._normalize_date(generated_query.end_date) or end_date
                self.logger.info(f"[{self.name}] LLM generated queries: {query_keywords}")

        start_date, end_date = self._validate_date_order(start_date, end_date)

        if not query_keywords:
            raise ValueError("没有可用的搜索关键词")

        # 用户未指定开始时间时，按提交时间倒序获取最新论文；否则按相关性排序
        if start_date is None:
            sort_by = arxiv.SortCriterion.SubmittedDate
            sort_order = arxiv.SortOrder.Descending
        else:
            sort_by = arxiv.SortCriterion.Relevance
            sort_order = arxiv.SortOrder.Descending

        papers = await self.paper_searcher.search_papers(
            querys=query_keywords,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            start_date=start_date,
            end_date=end_date,
        )

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
                "querys": query_keywords,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    async def _generate_search_queries(self, user_request: str) -> Optional[SearchQuery]:
        """使用 LLM Agent 根据用户请求生成检索查询条件。"""
        agent = get_search_agent()
        if agent is None:
            return None
        try:
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = textwrap.dedent(
                f"""\
                当前日期：{current_date}

                请根据用户查询需求，生成检索查询条件。
                用户查询需求：{user_request}
                """
            )
            response = await agent.run(task=prompt)
            content = response.messages[-1].content
            if isinstance(content, SearchQuery):
                return content
            if isinstance(content, str):
                return parse_search_query(content)
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


async def search_node(state: State) -> State:
    """LangGraph 适配函数：搜索节点。

    命令行阶段直接打印状态；前端接入时设置
    config={"search_node": {"frontend_enabled": True}} 即可切换为向 state_queue 推送消息。
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

        search_output = await node.run(search_input)
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
