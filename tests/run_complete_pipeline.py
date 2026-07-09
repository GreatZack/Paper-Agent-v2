"""
完整4节点Pipeline端到端测试：search → read(含verify) → parse → write
全部中间结果落盘保存到 output/complete_pipeline/
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.orchestrator import WorkflowOrchestrator
from src.core.state_models import BackToFrontData, ExecutionState
from src.nodes.write_node import WriteNode

USER_REQUEST = "帮我调研超声影像医学图像分割算法的近3年的进展"
MAX_PAPERS = 20
OUTPUT_DIR = Path("output/complete_pipeline")
ORCHESTRATOR_CONFIG = {
    "search_node": {
        "use_llm": True,
        "download_pdf": True,
        "pdf_download_dir": "data/papers",
    }
}


def save_json(filename: str, data) -> Path:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> saved {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    return path


def save_text(filename: str, text: str) -> Path:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> saved {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
    return path


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Paper-Agent-v2 完整 Pipeline 测试")
    print("=" * 60)
    print(f"  查询: {USER_REQUEST}")
    print(f"  最多论文: {MAX_PAPERS}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print()
    print("  ⏱ 预计耗时：20篇论文 + 验证循环，约 15-60 分钟，请耐心等待")
    print("    若遇到 API 限流或网络波动，耗时可能更长")
    print("=" * 60)
    print()

    state_queue = asyncio.Queue()
    orchestrator = WorkflowOrchestrator(
        state_queue=state_queue,
        config=ORCHESTRATOR_CONFIG,
    )

    workflow_task = asyncio.create_task(
        orchestrator.start(
            user_request=USER_REQUEST,
            max_papers=MAX_PAPERS,
        )
    )

    # 消费状态队列
    while True:
        try:
            message: BackToFrontData = await asyncio.wait_for(
                state_queue.get(), timeout=0.5
            )
            print(f"  [{message.step:>12}] {message.state:<12} {message.data or ''}")
            if message.step in ("completed", "failed", "stopped"):
                break
        except asyncio.TimeoutError:
            if workflow_task.done():
                break
            continue
        except asyncio.CancelledError:
            break

    final_state = await workflow_task

    print("\n" + "=" * 60)
    print("工作流结束，保存中间数据...")
    print("=" * 60)

    # 保存所有中间结果

    # 1. 搜索节点输出
    if final_state.search_output:
        save_json("01_search_results.json", final_state.search_output.model_dump())
        save_json(
            "02_search_papers_meta.json",
            [r.model_dump() for r in final_state.search_output.results],
        )

    # 2. 阅读节点输出
    read_data = {
        "key_info": [k.model_dump() for k in final_state.read_output.key_info],
        "failed_paper_ids": final_state.read_output.failed_paper_ids,
        "status": final_state.read_output.status,
    }
    save_json("03_read_keyinfo.json", read_data)

    verify_data = {
        pid: vr.model_dump()
        for pid, vr in (final_state.verify_results or {}).items()
    }
    save_json("04_read_verify.json", verify_data)

    # 3. 解析节点输出
    save_json("05_parse_output.json", final_state.parse_output.model_dump())

    # 4. 撰写节点输出
    if final_state.write_output and final_state.write_output.generated_text:
        save_text("06_write_report.md", final_state.write_output.generated_text)
    save_json("07_write_output.json", final_state.write_output.model_dump())

    # 5. 完整状态备份
    save_json("99_flow_state.json", final_state.model_dump())

    # 打印摘要
    print("\n" + "=" * 60)
    print("执行摘要")
    print("=" * 60)

    if final_state.current_step == ExecutionState.FAILED:
        print("\n❌ 工作流失败")
        err = final_state.error
        for node_name, node_err in [
            ("search_node", err.search_node_error),
            ("read_node", err.read_node_error),
            ("parse_node", err.parse_node_error),
            ("write_node", err.write_node_error),
        ]:
            if node_err:
                print(f"  失败节点: {node_name}")
                print(f"  错误信息: {node_err}")
    elif final_state.current_step == ExecutionState.STOPPED:
        print("\n⏹ 工作流被终止")
    else:
        print("\n✅ 工作流成功完成")

    print(f"\n  搜索论文数: {final_state.search_output.total_count if final_state.search_output else 0}")
    print(f"  提取成功: {len(final_state.read_output.key_info)}")
    print(f"  提取失败: {len(final_state.read_output.failed_paper_ids)}")
    print(f"  分类数量: {len(final_state.parse_output.taxonomy)}")
    print(f"  对比点数量: {len(final_state.parse_output.comparison_points)}")
    print(f"  矛盾点数量: {len(final_state.parse_output.discrepancies)}")
    if final_state.write_output and final_state.write_output.generated_text:
        report = final_state.write_output.generated_text
        print(f"  报告长度: {len(report)} 字符")
        print(f"  报告预览: {report[:100]}...")
    else:
        print(f"  报告长度: 0（未生成）")

    print(f"\n  所有输出文件: file://{OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
