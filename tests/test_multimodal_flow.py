"""端到端流程测试：search → read → parse，逐节点保存中间数据。

用法：
    python tests/test_multimodal_flow.py
    python tests/test_multimodal_flow.py --query "自定义查询" --max-papers 3

未传参时使用 main.py 中的 USER_REQUEST / MAX_PAPERS。
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import ORCHESTRATOR_CONFIG, USER_REQUEST, MAX_PAPERS
from src.graph.orchestrator import WorkflowOrchestrator
from src.core.state_models import BackToFrontData

OUTPUT_DIR = "output"


def save_json(filename: str, data) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> saved {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


def parse_args():
    parser = argparse.ArgumentParser(description="端到端流程测试：search → read → parse")
    parser.add_argument("--query", type=str, default=None, help="搜索查询（默认使用 main.py 中的 USER_REQUEST）")
    parser.add_argument("--max-papers", type=int, default=None, help="最大论文数（默认使用 main.py 中的 MAX_PAPERS）")
    return parser.parse_args()


async def main():
    args = parse_args()
    user_request = args.query or USER_REQUEST
    max_papers = args.max_papers or MAX_PAPERS

    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(
        state_queue=state_queue,
        config=ORCHESTRATOR_CONFIG,
    )

    print("=" * 60)
    print(f"启动工作流：search → read → parse")
    print(f"查询: {user_request}")
    print(f"最大论文数: {max_papers}")
    print("=" * 60)

    workflow_task = asyncio.create_task(
        orchestrator.start(
            user_request=user_request,
            max_papers=max_papers,
        )
    )

    # 消费状态队列
    while True:
        message: BackToFrontData = await state_queue.get()
        print(f"  [{message.step:>12}] {message.state:<12} {message.data or ''}")

        if message.step in ("completed", "failed", "stopped"):
            break

    print("\n" + "=" * 60)
    print("工作流结束，保存中间数据...")
    print("=" * 60)

    final_state = await workflow_task

    # 1. 搜索节点输出
    save_json("01_search_results.json", final_state.search_output.model_dump())
    save_json(
        "02_search_papers_meta.json",
        [r.model_dump() for r in final_state.search_output.results],
    )

    # 2. 阅读节点输出
    save_json(
        "03_read_keyinfo.json",
        [k.model_dump() for k in final_state.read_output.key_info],
    )
    save_json(
        "04_read_verify.json",
        {
            pid: vr.model_dump()
            for pid, vr in (final_state.verify_results or {}).items()
        },
    )
    save_json("04b_read_output.json", final_state.read_output.model_dump())

    # 3. 解析节点输出
    save_json("05_parse_output.json", final_state.parse_output.model_dump())

    # 4. 完整状态备份
    save_json("99_flow_state.json", final_state.model_dump())

    print("\n" + "=" * 60)
    print(f"全部输出已保存到 {OUTPUT_DIR}/ 目录")
    print("=" * 60)

    # 终端概览
    print(f"\n搜索论文数: {final_state.search_output.total_count}")
    print(f"提取成功: {len(final_state.read_output.key_info)}")
    print(f"提取失败: {len(final_state.read_output.failed_paper_ids)}")
    print(f"分类数量: {len(final_state.parse_output.taxonomy)}")
    print(f"对比点数量: {len(final_state.parse_output.comparison_points)}")
    print(f"矛盾点数量: {len(final_state.parse_output.discrepancies)}")


if __name__ == "__main__":
    asyncio.run(main())

