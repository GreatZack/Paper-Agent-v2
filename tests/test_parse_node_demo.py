"""用 read_node_demo 的真实数据测试 ParseNode 完整流程。"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.state_models import (
    KeyInformation,
    SearchResult,
    ParseInput,
)
from src.nodes.parse_node import ParseNode


def load_extraction(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def main():
    demo_dir = Path("output/read_node_demo")

    # ── 加载 demo 数据 ──
    extraction_files = sorted(demo_dir.glob("*_extraction.json"))
    print(f"找到 {len(extraction_files)} 篇论文的 extraction 数据\n")

    key_infos = []
    search_results = {}

    for path in extraction_files:
        data = load_extraction(path)
        paper_id = data["paper_id"]

        kw = KeyInformation(
            paper_id=paper_id,
            core_problem=data.get("core_problem", ""),
            key_methodology=data.get("key_methodology", ""),
            main_results=data.get("main_results", ""),
            limitations=data.get("limitations", ""),
            contributions=data.get("contributions", []),
            evidence_sections=data.get("evidence_sections", {}),
        )
        key_infos.append(kw)

        # SearchResult 演示：需要 title/authors
        sr = SearchResult(
            paper_id=paper_id,
            title=f"Paper {paper_id}",
            authors=[f"Author_{paper_id[:4]}"],
            summary="",
        )
        search_results[paper_id] = sr

    print(f"加载 {len(key_infos)} 篇论文的 KeyInformation")
    for kw in key_infos:
        print(f"  [{kw.paper_id}] core_problem: {kw.core_problem[:50]}...")

    # ── 构造 ParseInput ──
    parse_input = ParseInput(
        content=key_infos,
        paper_meta=search_results,
        parse_rules=[],
    )
    print(f"\nParseInput 构造完成: content={len(parse_input.content)} papers\n")

    # ── 运行 ParseNode ──
    node = ParseNode(config={})
    print("=" * 60)
    print("运行 ParseNode...")
    print("=" * 60)
    output = await node.process(parse_input)

    # ── 输出结果 ──
    print(f"\n状态: {output.status}")
    print(f"\n=== papers (清洗后论文数: {len(output.papers)}) ===")
    for pid, paper in output.papers.items():
        print(f"\n[{pid}] {paper.title}")
        print(f"  作者: {paper.authors}")
        print(f"  标签: {paper.tags}")
        print(f"  core_problem: {paper.core_problem[:60]}...")
        print(f"  key_methodology: {paper.key_methodology[:60]}...")
        print(f"  contributions: {len(paper.contributions)} 条")

    print(f"\n=== taxonomy (分类数: {len(output.taxonomy)}) ===")
    for cat, pids in output.taxonomy.items():
        print(f"  {cat}: {pids}")

    print(f"\n=== comparison_points (对比点数: {len(output.comparison_points)}) ===")
    for cp in output.comparison_points:
        print(f"  [{cp.topic}] comparable={cp.comparable}")
        for pid, entry in cp.entries.items():
            print(f"    {pid}: {entry}")
        if cp.note:
            print(f"    note: {cp.note}")

    print(f"\n=== discrepancies (矛盾数: {len(output.discrepancies)}) ===")
    for d in output.discrepancies:
        print(f"  [{d.topic}]")
        print(f"    A({d.paper_a_id}): {d.paper_a_claim[:60]}...")
        print(f"    B({d.paper_b_id}): {d.paper_b_claim[:60]}...")

    print(f"\n=== coverage ===")
    for dim, pids in output.coverage.items():
        print(f"  {dim}: {pids}")

    print(f"\n=== raw_extractions (原始备份数: {len(output.raw_extractions)}) ===")
    for pid, raw in output.raw_extractions.items():
        print(f"  [{pid}] evidence_sections: {list(raw.evidence_sections.keys())}")

    print("\n✅ 测试完成")


if __name__ == "__main__":
    asyncio.run(main())
