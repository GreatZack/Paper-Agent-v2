from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import arxiv
import httpx

from src.utils.log_utils import setup_logger

logger = setup_logger(__name__)


class PaperSearcher:
    """论文搜索器，使用 arxiv 库搜索论文。"""

    def __init__(self) -> None:
        pass

    async def search_papers(
        self,
        querys: List[str],
        max_results: int = 50,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
        sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """搜索 arXiv 论文。

        Args:
            querys: 搜索关键词列表。
            max_results: 最大返回结果数量。
            sort_by: 排序方式。
            sort_order: 排序顺序。
            start_date: 开始日期，字符串(YYYY-MM-DD)或 datetime 对象。
            end_date: 结束日期，字符串(YYYY-MM-DD)或 datetime 对象。

        Returns:
            论文信息字典列表。
        """
        try:
            if not querys:
                logger.warning("搜索关键词列表为空，跳过 arXiv 搜索")
                return []

            def _term(q: str) -> str:
                q = q.replace("\\", "\\\\").replace('"', '\\"')
                return f'all:"{q}"'

            search_query = "(" + " OR ".join(_term(q) for q in querys) + ")"

            if start_date or end_date:
                start_str = self._format_date(start_date) if start_date else "190001010000"
                end_str = (
                    self._format_date(end_date, end_of_day=True)
                    if end_date
                    else datetime.now(timezone.utc).strftime("%Y%m%d2359")
                )
                if start_str > end_str:
                    logger.warning(
                        f"开始日期 {start_date} 晚于结束日期 {end_date}，已自动交换"
                    )
                    start_str, end_str = end_str, start_str
                date_filter = f"submittedDate:[{start_str} TO {end_str}]"
                search_query += f" AND {date_filter}"

            logger.info(f"开始搜索论文: query='{search_query}', max_results={max_results}, sort_by={sort_by}")

            try:
                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
            except Exception as e:
                logger.error(f"创建 arxiv 搜索对象失败: {e}")
                return []

            client = arxiv.Client()
            papers = self.format_papers_list(client.results(search))
            logger.info(f"论文搜索完成，共找到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"论文搜索失败: {e}")
            raise

    async def search_by_topic(
        self,
        topic: str,
        limit: int = 10,
        recent_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """按主题搜索最近的论文。"""
        logger.info(f"按主题搜索论文: topic='{topic}', limit={limit}, recent_days={recent_days}")
        start_date = None
        if recent_days:
            start_date = datetime.now() - timedelta(days=recent_days)
        return await self.search_papers(
            querys=[topic],
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
            start_date=start_date,
        )

    def format_papers_list(self, search_results) -> List[Dict[str, Any]]:
        """将 arxiv 搜索结果格式化为论文信息字典列表。"""
        results_list = list(search_results)
        formatted_papers = [self._parse_paper_result(result) for result in results_list]
        logger.info(f"开始格式化论文列表，共 {len(results_list)} 篇论文")
        return formatted_papers

    async def download_pdf(
        self,
        pdf_url: str,
        paper_id: str,
        download_dir: str,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """异步下载 PDF 到本地目录。

        Args:
            pdf_url: PDF 下载链接。
            paper_id: 论文唯一标识，用于生成文件名。
            download_dir: 本地保存目录。
            timeout: 下载超时时间（秒）。

        Returns:
            下载成功返回本地文件路径，失败返回 None。
        """
        if not pdf_url:
            logger.warning(f"论文 {paper_id} 没有 pdf_url，跳过下载")
            return None

        safe_id = paper_id.replace("/", "_").replace("\\", "_")
        dir_path = Path(download_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / f"{safe_id}.pdf"

        if file_path.exists():
            logger.info(f"论文 {paper_id} 的 PDF 已存在: {file_path}")
            return str(file_path)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(pdf_url)
                response.raise_for_status()
                file_path.write_bytes(response.content)
            logger.info(f"论文 {paper_id} 的 PDF 下载完成: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.warning(f"下载论文 {paper_id} 的 PDF 失败: {e}")
            return None

    async def search_by_author(
        self,
        author_name: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """按作者搜索论文。"""
        logger.info(f"按作者搜索论文: author='{author_name}', limit={limit}")
        query = f"au:{author_name}"
        return await self.search_papers(
            querys=[query],
            max_results=limit,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

    def _parse_paper_result(self, result: arxiv.Result) -> Dict[str, Any]:
        """解析单条 arXiv 搜索结果。"""
        paper_id = result.get_short_id()
        published_year = result.published.year if result.published else None
        return {
            "paper_id": paper_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "published": published_year,
            "published_date": result.published.isoformat() if result.published else None,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "doi": result.doi if hasattr(result, "doi") else None,
        }

    def _format_date(
        self, date: Union[str, datetime], end_of_day: bool = False
    ) -> str:
        """格式化日期为 arXiv API 支持的格式 YYYYMMDD0000 或 YYYYMMDD2359。"""
        suffix = "2359" if end_of_day else "0000"
        if isinstance(date, datetime):
            return date.strftime(f"%Y%m%d{suffix}")
        elif isinstance(date, str):
            date_formats = [
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%Y.%m.%d",
                "%Y-%m",
                "%Y/%m",
                "%Y",
                "%Y年%m月%d日",
                "%Y年%m月",
                "%Y年",
            ]
            for fmt in date_formats:
                try:
                    if fmt == "%Y":
                        if len(date) == 4 and date.isdigit():
                            parsed_date = datetime(int(date), 1, 1)
                            return parsed_date.strftime(f"%Y%m%d{suffix}")
                    elif fmt in ["%Y-%m", "%Y/%m", "%Y年%m月"]:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.strftime(f"%Y%m%d{suffix}")
                    else:
                        parsed_date = datetime.strptime(date, fmt)
                        return parsed_date.strftime(f"%Y%m%d{suffix}")
                except ValueError:
                    continue
            try:
                from dateutil import parser
                parsed_date = parser.parse(date)
                return parsed_date.strftime(f"%Y%m%d{suffix}")
            except Exception:
                return datetime.now(timezone.utc).strftime(f"%Y%m%d{suffix}")
        return datetime.now(timezone.utc).strftime(f"%Y%m%d{suffix}")
