from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_pipeline_websocket_caps_requested_papers(monkeypatch):
    monkeypatch.setenv("MAX_PAPERS_LIMIT", "10")
    run_pipeline = AsyncMock()

    with (
        patch("backend.main.run_pipeline", run_pipeline),
        TestClient(app).websocket_connect("/ws/pipeline") as websocket,
    ):
        websocket.send_json({"query": "test query", "max_papers": 99})

    run_pipeline.assert_awaited_once()
    assert run_pipeline.await_args.args[:2] == ("test query", 10)


def test_pipeline_websocket_uses_configured_default(monkeypatch):
    monkeypatch.setenv("DEFAULT_MAX_PAPERS", "7")
    monkeypatch.setenv("MAX_PAPERS_LIMIT", "10")
    run_pipeline = AsyncMock()

    with (
        patch("backend.main.run_pipeline", run_pipeline),
        TestClient(app).websocket_connect("/ws/pipeline") as websocket,
    ):
        websocket.send_json({"query": "test query"})

    run_pipeline.assert_awaited_once()
    assert run_pipeline.await_args.args[:2] == ("test query", 7)
