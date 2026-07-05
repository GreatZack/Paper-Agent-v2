"""真实链路测试：ReadNode(2篇PDF) → ParseNode，输出保存到项目文件夹。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state_models import (
    ReadingStrategy,
    ReadInput,
    SearchResult,
    ParseInput,
)
from src.nodes.read_node import ReadNode
from src.nodes.parse_node import ParseNode

OUTPUT_DIR = Path("output/real_pipeline_test")


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 准备输入 ──
    papers_data = [
        SearchResult(
            paper_id="2607.02509v1",
            title="RECONTEXT: Training-Free Context Utilization in Long-Context Reasoning",
            authors=["Anonymous"],
            summary="",
            pdf_path="data/papers/2607.02509v1.pdf",
        ),
        SearchResult(
            paper_id="2607.02512v1",
            title="Program-as-Weights: Compiling Fuzzy Functions into Local Executable Neural Artifacts",
            authors=["Anonymous"],
            summary="",
            pdf_path="data/papers/2607.02512v1.pdf",
        ),
    ]

    # ── 2. 运行 ReadNode ──
    print("=" * 60)
    print("Step 1: ReadNode")
    print("=" * 60)
    read_node = ReadNode(config={"concurrency": 2, "max_tokens_threshold": 90000})
    read_input = ReadInput(
        documents=papers_data,
        reading_strategy=ReadingStrategy(
            focus_areas=["method", "result", "limitation"],
            depth="deep",
        ),
    )
    read_output = await read_node.process(read_input)

    # 保存 ReadNode 产出
    # 每篇论文的 KeyInformation
    for kw in read_output.key_info:
        path = OUTPUT_DIR / f"readnode_{kw.paper_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kw.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"  保存: {path}")
    # ReadOutput 汇总
    read_output_path = OUTPUT_DIR / "readnode_output_summary.json"
    with open(read_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": read_output.status,
            "key_info_count": len(read_output.key_info),
            "failed_paper_ids": read_output.failed_paper_ids,
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {read_output_path}")
    print(f"  read_output.status={read_output.status}, "
          f"成功={len(read_output.key_info)}, 失败={len(read_output.failed_paper_ids)}")

    # ── 3. 运行 ParseNode ──
    print("\n" + "=" * 60)
    print("Step 2: ParseNode")
    print("=" * 60)
    parse_node = ParseNode(config={})
    parse_input = ParseInput(
        content=read_output.key_info,
        paper_meta={p.paper_id: p for p in papers_data},
        parse_rules=[],
    )
    parse_output = await parse_node.process(parse_input)

    # 保存 ParseNode 产出
    # papers
    papers_path = OUTPUT_DIR / "parsenode_papers.json"
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump({
            pid: paper.model_dump() for pid, paper in parse_output.papers.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {papers_path}")

    # taxonomy
    taxonomy_path = OUTPUT_DIR / "parsenode_taxonomy.json"
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(parse_output.taxonomy, f, ensure_ascii=False, indent=2)
    print(f"  保存: {taxonomy_path}")

    # comparison_points
    cp_path = OUTPUT_DIR / "parsenode_comparison_points.json"
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump([cp.model_dump() for cp in parse_output.comparison_points],
                  f, ensure_ascii=False, indent=2)
    print(f"  保存: {cp_path}")

    # discrepancies
    disc_path = OUTPUT_DIR / "parsenode_discrepancies.json"
    with open(disc_path, "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in parse_output.discrepancies],
                  f, ensure_ascii=False, indent=2)
    print(f"  保存: {disc_path}")

    # coverage
    coverage_path = OUTPUT_DIR / "parsenode_coverage.json"
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(parse_output.coverage, f, ensure_ascii=False, indent=2)
    print(f"  保存: {coverage_path}")

    # raw_extractions (存成 summary，原文已在 readnode 中保存过)
    raw_path = OUTPUT_DIR / "parsenode_raw_extractions_summary.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({
            pid: {
                "paper_id": raw.paper_id,
                "evidence_sections": raw.evidence_sections,
            }
            for pid, raw in parse_output.raw_extractions.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {raw_path}")

    # ── 4. 打印摘要 ──
    print("\n" + "=" * 60)
    print("ParseNode 结果摘要")
    print("=" * 60)
    print(f"  状态: {parse_output.status}")
    print(f"  清洗后论文数: {len(parse_output.papers)}")
    for pid, paper in parse_output.papers.items():
        print(f"    [{pid}] tags={paper.tags}")
    print(f"  taxonomy: {parse_output.taxonomy}")
    print(f"  comparison_points: {len(parse_output.comparison_points)}")
    for cp in parse_output.comparison_points:
        print(f"    - {cp.topic}: comparable={cp.comparable}")
    print(f"  discrepancies: {len(parse_output.discrepancies)}")
    for d in parse_output.discrepancies:
        print(f"    - {d.topic}")
    print(f"  coverage: {dict(parse_output.coverage)}")

    print(f"\n所有输出文件保存在: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
