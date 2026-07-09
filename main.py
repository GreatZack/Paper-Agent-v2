import asyncio

from src.core.config import config
from src.core.state_models import BackToFrontData
from src.graph.orchestrator import WorkflowOrchestrator

# 导出为模块级常量，供测试脚本导入
USER_REQUEST = "帮我调研超声影像医学图像分割算法的近3年的进展"
MAX_PAPERS = int(config.get("default_max_papers", 50))
ORCHESTRATOR_CONFIG = {
    "search_node": {
        "use_llm": True,
        "download_pdf": True,
        "pdf_download_dir": "data/papers",
    }
}


async def main():
    """工作流示例入口。"""
    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(
        state_queue=state_queue,
        config=ORCHESTRATOR_CONFIG,
    )

    # 启动工作流（不阻塞，便于同时消费状态队列）
    workflow_task = asyncio.create_task(
        orchestrator.start(
            user_request=USER_REQUEST,
            max_papers=MAX_PAPERS,
        )
    )

    # 消费并打印状态队列
    while True:
        message: BackToFrontData = await state_queue.get()
        print(f"[{message.step}] {message.state}: {message.data}")
        if message.step in ("completed", "failed", "stopped"):
            break

    final_state = await workflow_task
    print("\nFinal state:")
    print(f"  current_step: {final_state.current_step}")
    print(f"  search_output.total_count: {final_state.search_output.total_count}")
    print(f"  read_output.key_info count: {len(final_state.read_output.key_info)}")
    print(f"  parse_output.papers count: {len(final_state.parse_output.papers)}")
    print(f"  parse_output.taxonomy: {list(final_state.parse_output.taxonomy.keys())}")
    print(f"  parse_output.comparison_points: {len(final_state.parse_output.comparison_points)}")
    print(f"  parse_output.discrepancies: {len(final_state.parse_output.discrepancies)}")
    print(f"  write_output.generated_text: {final_state.write_output.generated_text[:80]}...")

    print("\n搜索到的论文列表:")
    for idx, paper in enumerate(final_state.search_output.results, start=1):
        print(f"\n[{idx}] {paper.title}")
        print(f"    作者: {', '.join(paper.authors) if paper.authors else 'N/A'}")
        print(f"    发表时间: {paper.published or 'N/A'}")
        print(f"    链接: {paper.url}")
        print(f"    摘要: {paper.summary[:200]}...")


if __name__ == "__main__":
    asyncio.run(main())
