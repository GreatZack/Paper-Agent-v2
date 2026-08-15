"""FastAPI entry point with WebSocket endpoint for the paper-agent pipeline."""

import json
import logging
import os
import resource
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline_handler import run_pipeline
from src.core.config import config as app_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = FastAPI(title="Paper-Agent-v2 API")

# 简单的并发计数器（进程内，未线程隔离，仅用于运维观测）
_active_pipelines = 0
_counter_lock = threading.Lock()


def _positive_int_setting(name: str, default: int) -> int:
    """Read a positive integer environment setting with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logging.getLogger("config").warning(
            "Ignoring invalid %s=%r; using %d", name, raw, default
        )
        return default
    return value if value > 0 else default


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://greatzack.github.io",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """轻量运维观测：并发流水线数与进程 RSS。"""
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with _counter_lock:
        active = _active_pipelines
    return {
        "active_pipelines": active,
        "rss_mb": round(rss_mb, 1),
        "default_max_papers": int(app_config.get("default_max_papers", 10)),
    }


@app.websocket("/ws/pipeline")
async def pipeline_ws(websocket: WebSocket):
    global _active_pipelines
    await websocket.accept()
    logger = logging.getLogger("ws")

    try:
        data = await websocket.receive_json()
        query = data.get("query", "")
        raw = data.get("max_papers")
        configured_default = int(app_config.get("default_max_papers", 5))
        default_max_papers = _positive_int_setting(
            "DEFAULT_MAX_PAPERS", configured_default
        )
        max_papers_limit = _positive_int_setting("MAX_PAPERS_LIMIT", 10)
        requested_max_papers = int(raw) if raw else default_max_papers
        max_papers = max(1, min(requested_max_papers, max_papers_limit))

        if not query:
            await websocket.send_json(
                {
                    "step": "failed",
                    "state": "error",
                    "data": "query field is required",
                }
            )
            return

        logger.info("Starting pipeline: query=%s max_papers=%d", query, max_papers)
        with _counter_lock:
            _active_pipelines += 1
        try:
            await run_pipeline(query, max_papers, websocket)
        finally:
            with _counter_lock:
                _active_pipelines -= 1

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except json.JSONDecodeError:
        await websocket.send_json(
            {
                "step": "failed",
                "state": "error",
                "data": "Invalid JSON received",
            }
        )
    except Exception as exc:
        logger.exception("Pipeline error")
        try:
            await websocket.send_json(
                {
                    "step": "failed",
                    "state": "error",
                    "data": str(exc),
                }
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
