"""FastAPI entry point with WebSocket endpoint for the paper-agent pipeline."""
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline_handler import run_pipeline
from src.core.config import config as app_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

app = FastAPI(title="Paper-Agent-v2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/pipeline")
async def pipeline_ws(websocket: WebSocket):
    await websocket.accept()
    logger = logging.getLogger("ws")

    try:
        data = await websocket.receive_json()
        query = data.get("query", "")
        raw = data.get("max_papers")
        max_papers = int(raw) if raw else int(app_config.get("default_max_papers", 50))

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
        await run_pipeline(query, max_papers, websocket)

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
