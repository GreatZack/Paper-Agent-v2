"""全量链路测试：ReadNode(全部PDF) → ParseNode，输出保存到项目文件夹。"""

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

PDF_DIR = Path("data/papers")
OUTPUT_DIR = Path("output/full_pipeline_test")


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 扫描所有 PDF ──
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"找到 {len(pdf_files)} 篇 PDF\n")

    papers_data = []
    for pdf_path in pdf_files:
        paper_id = pdf_path.stem  # e.g. "2312.06351v1"
        papers_data.append(SearchResult(
            paper_id=paper_id,
            title=f"Paper {paper_id}",
            authors=[],
            summary="",
            pdf_path=str(pdf_path),
        ))
        print(f"  [{paper_id}] {pdf_path.name}")

    # ── 2. 运行 ReadNode ──
    print("\n" + "=" * 60)
    print("Step 1: ReadNode（8 篇 PDF，并发=3）")
    print("=" * 60)
    read_node = ReadNode(config={"concurrency": 3, "max_tokens_threshold": 90000})
    read_input = ReadInput(
        documents=papers_data,
        reading_strategy=ReadingStrategy(
            focus_areas=["method", "result", "limitation"],
            depth="deep",
        ),
    )
    read_output = await read_node.process(read_input)

    # 保存 ReadNode 产出
    read_dir = OUTPUT_DIR / "readnode"
    read_dir.mkdir(parents=True, exist_ok=True)

    for kw in read_output.key_info:
        path = read_dir / f"{kw.paper_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kw.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"  保存: {path}")

    # 失败的论文记录
    fail_path = read_dir / "_failed_papers.json"
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(read_output.failed_paper_ids, f, ensure_ascii=False, indent=2)
    print(f"  保存: {fail_path}")

    # 汇总
    summary_path = read_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "status": read_output.status,
            "total": len(read_output.key_info) + len(read_output.failed_paper_ids),
            "success": len(read_output.key_info),
            "failed": len(read_output.failed_paper_ids),
            "failed_ids": read_output.failed_paper_ids,
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {summary_path}")
    print(f"  → 成功={len(read_output.key_info)}, 失败={len(read_output.failed_paper_ids)}")

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
    parse_dir = OUTPUT_DIR / "parsenode"
    parse_dir.mkdir(parents=True, exist_ok=True)

    # papers
    with open(parse_dir / "papers.json", "w", encoding="utf-8") as f:
        json.dump({
            pid: paper.model_dump() for pid, paper in parse_output.papers.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'papers.json'}")

    # taxonomy
    with open(parse_dir / "taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(parse_output.taxonomy, f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'taxonomy.json'}")

    # comparison_points
    with open(parse_dir / "comparison_points.json", "w", encoding="utf-8") as f:
        json.dump([cp.model_dump() for cp in parse_output.comparison_points],
                  f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'comparison_points.json'}")

    # discrepancies
    with open(parse_dir / "discrepancies.json", "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in parse_output.discrepancies],
                  f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'discrepancies.json'}")

    # coverage
    with open(parse_dir / "coverage.json", "w", encoding="utf-8") as f:
        json.dump(parse_output.coverage, f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'coverage.json'}")

    # raw_extractions 摘要
    with open(parse_dir / "raw_extractions_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            pid: {
                "paper_id": raw.paper_id,
                "evidence_sections": raw.evidence_sections,
            }
            for pid, raw in parse_output.raw_extractions.items()
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {parse_dir / 'raw_extractions_summary.json'}")

    # ── 4. 打印摘要 ──
    print("\n" + "=" * 60)
    print("ParseNode 结果摘要")
    print("=" * 60)
    print(f"  状态: {parse_output.status}")
    print(f"  论文数: {len(parse_output.papers)}")
    for pid, paper in parse_output.papers.items():
        print(f"    [{pid}] tags={paper.tags}")
    print(f"  taxonomy:")
    for cat, pids in parse_output.taxonomy.items():
        print(f"    {cat}: {pids}")
    print(f"  comparison_points: {len(parse_output.comparison_points)}")
    for cp in parse_output.comparison_points:
        print(f"    - {cp.topic}: comparable={cp.comparable} entries={len(cp.entries)}")
        if cp.note:
            print(f"      note: {cp.note}")
    print(f"  discrepancies: {len(parse_output.discrepancies)}")
    for d in parse_output.discrepancies:
        print(f"    - {d.topic}: {d.paper_a_id} vs {d.paper_b_id}")
    print(f"  coverage: {dict(parse_output.coverage)}")

    print(f"\n所有输出文件: file://{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
