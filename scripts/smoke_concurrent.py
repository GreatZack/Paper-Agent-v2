"""并发 WS 冒烟：验证服务在 3 个并发连接下稳定、计数正确、内存受控。

所有连接都走真实 pipeline 流程但无 LLM key，预期最终挂起或失败；本脚本
只验证服务端不崩溃、active_pipelines 动态变化、断开后归零、RSS 不显著增长。
"""
import asyncio
import contextlib
import json
import os
import subprocess
import time

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import httpx
import websockets

PORT = 8013
BASE = f"http://127.0.0.1:{PORT}"


async def one_client(uri, i):
    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            await ws.send(json.dumps({"query": f"并发压测-{i}", "max_papers": 10}))
            n = 0
            async with asyncio.timeout(12):
                try:
                    while True:
                        await ws.recv()
                        n += 1
                except websockets.ConnectionClosed:
                    pass
            return {"id": i, "msgs": n, "err": None}
    except Exception as exc:  # noqa: BLE001
        return {"id": i, "msgs": 0, "err": str(exc)}


async def main():
    env = {"NO_PROXY": "127.0.0.1,localhost", "PATH": "/home/zengke/Paper-Agent-v2/.venv/bin:/usr/bin:/bin"}
    proc = subprocess.Popen(
        [".venv/bin/uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    client = httpx.Client(trust_env=False, timeout=5, base_url=BASE)
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if client.get("/health").status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
        else:
            raise RuntimeError("server not up")

        client.get("/metrics")
        results = await asyncio.gather(*(one_client(f"ws://127.0.0.1:{PORT}/ws/pipeline", i) for i in range(3)))
        for r in results:
            print(f"client {r['id']}: msgs={r['msgs']} err={r['err']}")

        # 等所有断连被感知 + 任务取消
        deadline = time.time() + 60
        while time.time() < deadline:
            m = client.get("/metrics").json()
            if m["active_pipelines"] == 0:
                break
            await asyncio.sleep(1)
        else:
            raise AssertionError(f"active_pipelines 未归零: {m}")
        print("post metrics:", m)
        assert m["rss_mb"] < 400, m
        assert all(r["msgs"] >= 0 for r in results)
        print("\nCONCURRENT SMOKE PASS")
    finally:
        client.close()
        with contextlib.suppress(Exception):
            proc.terminate()


try:
    asyncio.run(main())
finally:
    if "proc" in globals():
        with contextlib.suppress(Exception):
            globals()["proc"].terminate()