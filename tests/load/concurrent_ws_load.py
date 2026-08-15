"""真实并发 WebSocket 压测：验证多人同时运行 10 篇论文流水线的稳定性。

用法（需先起后端）:
    uvicorn backend.main:app --port 8000
    python tests/load/concurrent_ws_load.py --clients 3 --max-papers 10 --duration 600

将断言：
- 每个连接最终收到 report 或失败消息（不挂死）
- 服务未因内存 OOM 断开（无异常关闭单个连接）
- 打印各连接墙钟 + 进程指标峰值
"""

import argparse
import asyncio
import json
import time
from typing import Dict, Optional

import httpx
import websockets


async def run_one_client(
    uri: str,
    query: str,
    max_papers: int,
    results: Dict[str, object],
    client_id: str,
) -> None:
    start = time.perf_counter()
    try:
        async with websockets.connect(
            uri, open_timeout=30, close_timeout=10, ping_interval=30
        ) as ws:
            await ws.send(json.dumps({"query": query, "max_papers": max_papers}))
            while True:
                raw = await ws.recv()
                msg = json.loads(raw)
                if msg.get("step") in ("report", "failed", "stopped"):
                    results[client_id] = {
                        "step": msg.get("step"),
                        "wall_s": round(time.perf_counter() - start, 1),
                        "report_len": len(msg.get("data") or ""),
                    }
                    return
    except Exception as exc:  # noqa: BLE001
        results[client_id] = {
            "step": "connection_error",
            "wall_s": round(time.perf_counter() - start, 1),
            "error": str(exc),
        }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws/pipeline")
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--max-papers", type=int, default=10)
    parser.add_argument("--query", default="超声影像医学图像分割算法的最新进展")
    parser.add_argument("--metrics", default="http://127.0.0.1:8000/metrics")
    args = parser.parse_args()

    results: Dict[str, object] = {}
    tasks = [
        asyncio.create_task(
            run_one_client(args.uri, args.query, args.max_papers, results, f"cli-{i}")
        )
        for i in range(args.clients)
    ]

    t0 = time.perf_counter()
    await asyncio.gather(*tasks)
    total = time.perf_counter() - t0

    print(f"\n并发 {args.clients} 连接 × max_papers={args.max_papers}，总耗时 {total:.1f}s")
    ok = 0
    for cid, r in results.items():
        print(f"  {cid}: {r}")
        if r.get("step") == "report":
            ok += 1
    print(f"成功返回 report 数: {ok}/{args.clients}")

    # 拉一次 /metrics 看进程 RSS 与并发计数（仅供参考）
    async with httpx.AsyncClient(timeout=10) as hc:
        try:
            r = await hc.get(args.metrics)
            print("服务 /metrics:", r.json())
        except Exception as exc:  # noqa: BLE001
            print("获取 /metrics 失败:", exc)

    # 判定：所有连接必须正常终止（report 或业务 failed/stopped），无 connection_error
    errors = [c for c, r in results.items() if r.get("step") == "connection_error"]
    if errors:
        print(f"\n[FAIL] {len(errors)} 个连接异常中断，疑似 OOM 或服务重启。")
        raise SystemExit(1)
    print(f"\n[PASS] {ok}/{args.clients} 成功；无连接异常中断。")


if __name__ == "__main__":
    asyncio.run(main())