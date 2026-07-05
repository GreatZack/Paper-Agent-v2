import asyncio

from src.core.state_models import BackToFrontData
from src.graph.orchestrator import WorkflowOrchestrator


async def main():
    """工作流示例入口。"""
    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(
        state_queue=state_queue,
        config={
            "search_node": {
                "use_llm": True,
                "download_pdf": True,
                "pdf_download_dir": "data/papers",
            }
        },
    )

    # 启动工作流（不阻塞，便于同时消费状态队列）
    workflow_task = asyncio.create_task(
        orchestrator.start(
            user_request="帮我调研大型语言模型在自动驾驶领域的应用现状",
            max_papers=3,
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
    print(f"  parse_output.structured_data: {final_state.parse_output.structured_data}")
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
