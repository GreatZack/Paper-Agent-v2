from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tasks.paper_search import PaperSearcher


class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeResult:
    def __init__(self, **kwargs):
        self.paper_id = kwargs.get("paper_id", "1234.56789")
        self.title = kwargs.get("title", "Test Paper")
        self.authors = [FakeAuthor(a) for a in kwargs.get("authors", ["Alice"])]
        self.summary = kwargs.get("summary", "Summary")
        self.published = kwargs.get("published", datetime(2024, 1, 1))
        self.entry_id = kwargs.get("entry_id", "http://arxiv.org/abs/1234.56789")
        self.pdf_url = kwargs.get("pdf_url", "http://arxiv.org/pdf/1234.56789")
        self.primary_category = kwargs.get("primary_category", "cs.AI")
        self.categories = kwargs.get("categories", ["cs.AI"])
        self.doi = kwargs.get("doi", None)

    def get_short_id(self):
        return self.paper_id


def _make_mock_client(results_list):
    """构造一个模拟的 arxiv.Client，返回指定的结果列表。"""
    fake_client = MagicMock()
    fake_client.results.return_value = results_list
    return fake_client


@pytest.fixture
def searcher():
    return PaperSearcher()


@pytest.mark.asyncio
async def test_format_date_variants(searcher):
    assert searcher._format_date("2023") == "202301010000"
    assert searcher._format_date("2023-05") == "202305010000"
    assert searcher._format_date("2023-05-15") == "202305150000"
    assert searcher._format_date("2023/05/15") == "202305150000"
    assert searcher._format_date(datetime(2023, 5, 15)) == "202305150000"


@pytest.mark.asyncio
async def test_parse_paper_result(searcher):
    result = FakeResult(
        paper_id="2411.11607v2",
        title="Performance evaluation",
        authors=["Bob", "Carol"],
        summary="A summary",
        published=datetime(2024, 11, 18, 14, 29, 22),
        entry_id="http://arxiv.org/abs/2411.11607v2",
        pdf_url="http://arxiv.org/pdf/2411.11607v2",
        primary_category="cs.RO",
        categories=["cs.RO"],
        doi="10.5220/0012556800003702",
    )
    paper = searcher._parse_paper_result(result)
    assert paper["paper_id"] == "2411.11607v2"
    assert paper["title"] == "Performance evaluation"
    assert paper["authors"] == ["Bob", "Carol"]
    assert paper["published"] == 2024
    assert paper["published_date"] == "2024-11-18T14:29:22"
    assert paper["url"] == "http://arxiv.org/abs/2411.11607v2"
    assert paper["pdf_url"] == "http://arxiv.org/pdf/2411.11607v2"
    assert paper["primary_category"] == "cs.RO"
    assert paper["doi"] == "10.5220/0012556800003702"


@pytest.mark.asyncio
async def test_search_papers_success(searcher):
    fake_result = FakeResult(
        paper_id="2411.11607v2",
        title="ROS2 Automated Driving",
        authors=["Alice"],
        summary="Summary text",
        published=datetime(2024, 11, 18),
    )
    fake_client = _make_mock_client([fake_result])

    with patch("src.tasks.paper_search.arxiv.Client", return_value=fake_client):
        papers = await searcher.search_papers(querys=["ROS2", "automated driving"], max_results=5)

    assert len(papers) == 1
    assert papers[0]["paper_id"] == "2411.11607v2"
    fake_client.results.assert_called_once()


@pytest.mark.asyncio
async def test_search_papers_with_date_range(searcher):
    fake_result = FakeResult(paper_id="2307.06258v1", title="Safe AD", published=datetime(2023, 7, 12))
    fake_client = _make_mock_client([fake_result])

    with patch("src.tasks.paper_search.arxiv.Client", return_value=fake_client) as mock_client_cls:
        papers = await searcher.search_papers(
            querys=["automated driving"],
            max_results=10,
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

    assert len(papers) == 1
    call_args = mock_client_cls.call_args
    assert call_args is not None


@pytest.mark.asyncio
async def test_search_papers_empty_results(searcher):
    fake_client = _make_mock_client([])

    with patch("src.tasks.paper_search.arxiv.Client", return_value=fake_client):
        papers = await searcher.search_papers(querys=["nonexistent topic"], max_results=10)

    assert papers == []


@pytest.mark.asyncio
async def test_search_papers_search_object_failure(searcher):
    with patch("src.tasks.paper_search.arxiv.Search", side_effect=Exception("arxiv error")):
        papers = await searcher.search_papers(querys=["test"], max_results=10)
    assert papers == []


@pytest.mark.asyncio
async def test_search_by_topic(searcher):
    fake_result = FakeResult(paper_id="1234.56789", title="Topic Paper", published=datetime(2024, 1, 1))
    fake_client = _make_mock_client([fake_result])

    with patch("src.tasks.paper_search.arxiv.Client", return_value=fake_client):
        papers = await searcher.search_by_topic(topic="LLM", limit=5, recent_days=30)

    assert len(papers) == 1
    assert papers[0]["title"] == "Topic Paper"


@pytest.mark.asyncio
async def test_search_by_author(searcher):
    fake_result = FakeResult(paper_id="1234.56789", title="Author Paper", published=datetime(2024, 1, 1))
    fake_client = _make_mock_client([fake_result])

    with patch("src.tasks.paper_search.arxiv.Client", return_value=fake_client):
        papers = await searcher.search_by_author(author_name="Geoffrey Hinton", limit=5)

    assert len(papers) == 1
    assert papers[0]["title"] == "Author Paper"


@pytest.mark.asyncio
async def test_download_pdf_success(searcher, tmp_path):
    pdf_url = "http://arxiv.org/pdf/1234.56789"
    paper_id = "1234.56789"
    download_dir = str(tmp_path / "papers")

    fake_response = MagicMock()
    fake_response.content = b"fake pdf content"
    fake_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=fake_response):
        pdf_path = await searcher.download_pdf(pdf_url, paper_id, download_dir)

    assert pdf_path is not None
    assert pdf_path.endswith("1234.56789.pdf")


@pytest.mark.asyncio
async def test_download_pdf_skip_existing(searcher, tmp_path):
    pdf_url = "http://arxiv.org/pdf/1234.56789"
    paper_id = "1234.56789"
    download_dir = tmp_path / "papers"
    download_dir.mkdir(parents=True, exist_ok=True)
    existing_file = download_dir / "1234.56789.pdf"
    existing_file.write_text("already exists")

    pdf_path = await searcher.download_pdf(pdf_url, paper_id, str(download_dir))
    assert pdf_path == str(existing_file)
