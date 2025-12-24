"""API contract tests for /chat and /ingest endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.api.main import create_app
from backend.data.chat_repo import InMemoryChatRepository
from backend.data.config import Settings, VectorBackend
from backend.orchestrator.state import GraphState, RouteDecision


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        tavily_api_key="tvly-test",
        mongodb_uri="memory://",
        feature_vector_backend=VectorBackend.faiss,
        faiss_index_path="/tmp/test_faiss",
        langsmith_api_key=None,
        langsmith_tracing=False,
    )


@pytest.fixture
def test_app(test_settings: Settings) -> Any:
    app = create_app()
    return app


@pytest.fixture
def mock_chat_repo() -> InMemoryChatRepository:
    return InMemoryChatRepository()


@pytest.fixture
def mock_vector_store() -> Any:
    vs = AsyncMock()
    vs.similarity_search = AsyncMock(return_value=[])
    vs.upsert = AsyncMock(return_value=5)
    return vs


@pytest.mark.asyncio
async def test_health_endpoint(test_app: Any) -> None:
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_chat_creates_session(
    test_app: Any,
    test_settings: Settings,
    mock_vector_store: Any,
    mock_chat_repo: InMemoryChatRepository,
) -> None:
    final_state = GraphState(
        question="What is RAG?",
        session_id="new-session-123",
        answer="RAG stands for Retrieval-Augmented Generation.",
        route=RouteDecision.local_rag,
        confidence=0.85,
        sources=[],
        latency_ms=120.0,
        token_usage={"input_tokens": 10, "output_tokens": 20},
    )

    with (
        patch("backend.api.routes.run_pipeline", return_value=final_state),
        patch("backend.api.dependencies.get_settings", return_value=test_settings),
        patch("backend.api.dependencies.get_vector_store", return_value=mock_vector_store),
        patch("backend.api.dependencies.get_chat_repo", return_value=mock_chat_repo),
    ):
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            resp = await client.post("/chat", json={"question": "What is RAG?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "answer" in data
    assert "route" in data
    assert data["route"] == "LOCAL_RAG"


@pytest.mark.asyncio
async def test_chat_missing_question(test_app: Any) -> None:
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        resp = await client.post("/chat", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_endpoint(
    test_app: Any,
    test_settings: Settings,
    mock_vector_store: Any,
) -> None:
    with (
        patch("backend.api.routes.ingest_file", new_callable=AsyncMock, return_value=10),
        patch("backend.api.dependencies.get_settings", return_value=test_settings),
        patch("backend.api.dependencies.get_vector_store", return_value=mock_vector_store),
    ):
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            resp = await client.post("/ingest", json={"file_path": "/data/test.pdf"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunks_stored"] == 10


@pytest.mark.asyncio
async def test_ingest_invalid_file_type(
    test_app: Any,
    test_settings: Settings,
    mock_vector_store: Any,
) -> None:
    with (
        patch("backend.api.routes.ingest_file", side_effect=ValueError("Unsupported file type: .xyz")),
        patch("backend.api.dependencies.get_settings", return_value=test_settings),
        patch("backend.api.dependencies.get_vector_store", return_value=mock_vector_store),
    ):
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            resp = await client.post("/ingest", json={"file_path": "/data/test.xyz"})

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_get_chat_history_not_found(
    test_app: Any,
    test_settings: Settings,
    mock_chat_repo: InMemoryChatRepository,
) -> None:
    with (
        patch("backend.api.dependencies.get_settings", return_value=test_settings),
        patch("backend.api.dependencies.get_chat_repo", return_value=mock_chat_repo),
    ):
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            resp = await client.get("/chat/nonexistent-session-id")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_chat_history_found(
    test_app: Any,
    test_settings: Settings,
    mock_chat_repo: InMemoryChatRepository,
) -> None:
    session = await mock_chat_repo.create_session()
    from backend.data.chat_repo import ChatMessage

    await mock_chat_repo.append_messages(
        session.session_id,
        [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="Hi!")],
    )

    with (
        patch("backend.api.dependencies.get_settings", return_value=test_settings),
        patch("backend.api.dependencies.get_chat_repo", return_value=mock_chat_repo),
    ):
        async with AsyncClient(app=test_app, base_url="http://test") as client:
            resp = await client.get(f"/chat/{session.session_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
