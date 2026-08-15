"""一次启动 uvicorn 并跑完整 WS 冒烟验证。用法: python scripts/smoke_ws.py"""
import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

import httpx
import websockets

PORT = 8012
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/ws/pipeline"

proc = None


async def main():
    global proc
    env = {"NO_PROXY": "127.0.0.1,localhost", "PATH": "/home/zengke/Paper-Agent-v2/.venv/bin:" + "/usr/bin:/bin"}
    proc = subprocess.Popen(
        [".venv/bin/uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 30
    client = httpx.Client(trust_env=False, timeout=5, base_url=BASE)
    ok = False
    while time.time() < deadline:
        try:
            r = client.get("/health")
            if r.status_code == 200:
                ok = True
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    assert ok, f"server not up: {proc.returncode}"
    print("health OK:", r.json())

    m = client.get("/metrics").json()
    print("metrics:", m)
    assert m["default_max_papers"] == 10, m
    assert m["rss_mb"] > 0

    t0 = time.perf_counter()
    async with websockets.connect(WS, open_timeout=10) as ws:
        await ws.send(json.dumps({"query": "冒烟测试", "max_papers": 10}))
        got = None
        while time.time() - t0 < 45:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=45)
            except asyncio.TimeoutError:
                print("(no message 45s)")
                break
            msg = json.loads(raw)
            print(f"{(time.perf_counter()-t0)*1000:.0f}ms step={msg.get('step')} state={msg.get('state')}")
            if msg.get("step") in ("report", "failed", "stopped"):
                got = msg
                break
        if got is None:
            print("(no terminal message; likely no API key configured)")
        else:
            print("final data:", str(got.get("data"))[:200])
            assert "cannot access local variable" not in str(got.get("data")), f"bug: {got.get('data')}"

    # 客户端断开后服务端需感知断连（下一轮 send/heartbeat 失败）才能取消任务并归零计数
    deadline = time.time() + 45
    while time.time() < deadline:
        m = client.get("/metrics").json()
        if m["active_pipelines"] == 0:
            break
        await asyncio.sleep(1)
    else:
        raise AssertionError(f"active_pipelines 未归零: {m}")
    print("post metrics:", m)
    assert m["active_pipelines"] == 0, m
    client.close()
    print("\nSMOKE TEST PASS")


try:
    asyncio.run(main())
finally:
    if proc:
        with contextlib.suppress(Exception):
            proc.terminate()