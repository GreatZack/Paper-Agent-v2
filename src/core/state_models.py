from asyncio import Queue
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ExecutionState(str, Enum):
    """工作流执行状态枚举"""

    INITIALIZING = "initializing"
    SEARCHING = "searching"
    READING = "reading"
    PARSING = "parsing"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    STOPPED = "stopped"


class BackToFrontData(BaseModel):
    """推送给前端的状态更新数据"""

    step: str
    state: str
    data: Any = None


# ==================== 搜索节点数据模型 ====================


class SearchScope(BaseModel):
    """搜索范围配置"""

    sources: List[str] = Field(default_factory=list, description="搜索来源列表，如 arxiv、google_scholar")
    start_date: Optional[str] = Field(default=None, description="开始时间, 格式: YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="结束时间, 格式: YYYY-MM-DD")
    max_results: int = Field(default=10, description="最大返回结果数")


class SearchInput(BaseModel):
    """搜索节点输入"""

    query_keywords: List[str] = Field(default_factory=list, description="查询关键词列表")
    search_scope: SearchScope = Field(default_factory=SearchScope, description="搜索范围")
    user_request: str = Field(default="", description="用户原始请求")


class SearchResult(BaseModel):
    """单条搜索结果"""

    paper_id: str = Field(default="", description="论文/文档唯一标识")
    title: str = Field(default="", description="标题")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    summary: str = Field(default="", description="摘要/简介")
    url: str = Field(default="", description="访问链接")
    pdf_url: Optional[str] = Field(default=None, description="PDF 下载链接")
    pdf_path: Optional[str] = Field(default=None, description="本地 PDF 文件路径")
    published: Optional[str] = Field(default=None, description="发布时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class SearchOutput(BaseModel):
    """搜索节点输出"""

    results: List[SearchResult] = Field(default_factory=list, description="搜索结果列表")
    total_count: int = Field(default=0, description="结果总数")
    status: str = Field(default="pending", description="搜索状态: pending/running/completed/failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="搜索元数据")


# ==================== 阅读节点数据模型 ====================


class ReadingStrategy(BaseModel):
    """阅读策略配置"""

    focus_areas: List[str] = Field(default_factory=list, description="关注维度，如 [method, result, limitation]")
    depth: str = Field(default="medium", description="阅读深度: shallow/medium/deep")
    extract_questions: List[str] = Field(default_factory=list, description="引导性问题列表")


class ReadInput(BaseModel):
    """阅读节点输入"""

    documents: List[SearchResult] = Field(default_factory=list, description="待阅读的文档列表")
    reading_strategy: ReadingStrategy = Field(default_factory=ReadingStrategy, description="阅读策略")


class KeyInformation(BaseModel):
    """单篇文档提取出的关键信息"""

    paper_id: str = Field(default="", description="关联文档ID")
    core_problem: str = Field(default="", description="核心问题")
    key_methodology: str = Field(default="", description="关键方法")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default_factory=list, description="贡献列表")
    evidence_sections: Dict[str, str] = Field(default_factory=dict, description="各字段对应的原文章节引用")


class ReadOutput(BaseModel):
    """阅读节点输出"""

    key_info: List[KeyInformation] = Field(default_factory=list, description="关键信息提取结果")
    status: str = Field(default="pending", description="阅读状态")
    failed_paper_ids: List[str] = Field(default_factory=list, description="阅读失败的文档ID")


# ==================== 解析节点数据模型 ====================


class ParseRule(BaseModel):
    """单条解析规则"""

    name: str = Field(default="", description="规则名称")
    field_path: str = Field(default="", description="目标字段路径")
    data_type: str = Field(default="string", description="期望数据类型")
    required: bool = Field(default=False, description="是否必填")


class ParsedPaper(BaseModel):
    """清洗归一化后的单篇论文信息"""

    paper_id: str = Field(default="", description="论文唯一标识")
    title: str = Field(default="", description="论文标题")
    authors: str = Field(default="", description="作者列表（逗号分隔）")
    core_problem: str = Field(default="", description="核心问题")
    key_methodology: str = Field(default="", description="关键方法")
    main_results: str = Field(default="", description="主要结果")
    limitations: str = Field(default="", description="局限性")
    contributions: List[str] = Field(default_factory=list, description="贡献列表")
    tags: List[str] = Field(default_factory=list, description="分类标签")


class ComparisonPoint(BaseModel):
    """跨论文对比数据点"""

    topic: str = Field(default="", description="对比主题，如 'CIFAR-10 Top-1 Acc'")
    comparable: bool = Field(default=False, description="是否可直接对比")
    entries: Dict[str, str] = Field(default_factory=dict, description="paper_id → 数值/描述")
    note: Optional[str] = Field(default=None, description="不可比时的说明")


class DiscrepancyNote(BaseModel):
    """论文间的矛盾点"""

    topic: str = Field(default="", description="冲突主题")
    paper_a_id: str = Field(default="", description="论文A的ID")
    paper_a_claim: str = Field(default="", description="论文A的结论")
    paper_b_id: str = Field(default="", description="论文B的ID")
    paper_b_claim: str = Field(default="", description="论文B的结论")


class ParseInput(BaseModel):
    """解析节点输入"""

    content: List[KeyInformation] = Field(default_factory=list, description="待解析内容")
    paper_meta: Dict[str, SearchResult] = Field(
        default_factory=dict,
        description="paper_id → SearchResult，提供 title/authors/pdf_path 等元信息",
    )
    parse_rules: List[ParseRule] = Field(default_factory=list, description="解析规则")


class ParseOutput(BaseModel):
    """解析节点输出"""

    papers: Dict[str, ParsedPaper] = Field(
        default_factory=dict,
        description="清洗后的论文数据，按 paper_id 索引",
    )
    raw_extractions: Dict[str, KeyInformation] = Field(
        default_factory=dict,
        description="原始 KeyInformation 备份，供下游回查",
    )
    taxonomy: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="方法分类 → 论文 ID 列表",
    )
    comparison_points: List[ComparisonPoint] = Field(
        default_factory=list,
        description="跨论文可对比的数据点",
    )
    discrepancies: List[DiscrepancyNote] = Field(
        default_factory=list,
        description="论文间的矛盾或结果差异",
    )
    coverage: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="每个维度被哪些论文覆盖，如 {'core_problem': ['id1', 'id2']}",
    )
    status: str = Field(default="pending", description="解析状态")


# ==================== 撰写节点数据模型 ====================


class WriteInput(BaseModel):
    """撰写节点输入"""

    structured_data: Dict[str, Any] = Field(default_factory=dict, description="结构化数据")
    template: str = Field(default="", description="撰写模板")
    style_hints: Dict[str, Any] = Field(default_factory=dict, description="风格/格式提示")


class WriteOutput(BaseModel):
    """撰写节点输出"""

    generated_text: str = Field(default="", description="生成文本")
    status: str = Field(default="pending", description="撰写状态")
    section_map: Dict[str, str] = Field(default_factory=dict, description="章节映射")


# ==================== 错误与全局状态模型 ====================


class NodeError(BaseModel):
    """各节点错误信息"""

    search_node_error: Optional[str] = Field(default=None, description="搜索节点错误信息")
    read_node_error: Optional[str] = Field(default=None, description="阅读节点错误信息")
    parse_node_error: Optional[str] = Field(default=None, description="解析节点错误信息")
    write_node_error: Optional[str] = Field(default=None, description="撰写节点错误信息")
    error: Optional[str] = Field(default=None, description="通用错误信息")


class PaperAgentState(BaseModel):
    """LangGraph 工作流全局状态对象"""

    # 用户输入
    user_request: str = Field(description="用户的原始输入请求")
    max_papers: int = Field(default=50, description="最大论文数量")

    # 执行状态
    current_step: ExecutionState = Field(default=ExecutionState.INITIALIZING, description="当前执行步骤")
    error: NodeError = Field(default_factory=NodeError, description="错误信息")

    # 数据流
    search_output: SearchOutput = Field(default_factory=SearchOutput, description="搜索节点输出")
    read_output: ReadOutput = Field(default_factory=ReadOutput, description="阅读节点输出")
    parse_output: ParseOutput = Field(default_factory=ParseOutput, description="解析节点输出")
    write_output: WriteOutput = Field(default_factory=WriteOutput, description="撰写节点输出")

    # 配置与上下文
    config: Dict[str, Any] = Field(default_factory=dict, description="运行时配置")
    agent_logs: Dict[str, str] = Field(default_factory=dict, description="各智能体执行日志，key为智能体名称")
    frontend_data: Optional[BackToFrontData] = Field(default=None, description="前端展示数据")


class State(TypedDict):
    """LangGraph 兼容的状态定义"""

    state_queue: Queue
    value: PaperAgentState


class ConfigSchema(TypedDict):
    """LangGraph 兼容的配置定义"""

    state_queue: Queue
    value: Dict[str, Any]
