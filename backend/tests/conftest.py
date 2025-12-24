"""Shared fixtures for the test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.data.chat_repo import InMemoryChatRepository
from backend.data.config import Settings, VectorBackend
from backend.orchestrator.state import GraphState, RouteDecision


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        openai_api_key="sk-test",
        tavily_api_key="tvly-test",
        mongodb_uri="memory://",
        mongodb_db_name="test_db",
        qdrant_url="http://localhost:6333",
        feature_vector_backend=VectorBackend.faiss,
        faiss_index_path="/tmp/test_faiss",
        langsmith_api_key=None,
        langsmith_tracing=False,
        confidence_threshold=0.7,
        retrieval_top_k=3,
    )


@pytest.fixture
def chat_repo() -> InMemoryChatRepository:
    return InMemoryChatRepository()


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Retrieval-augmented generation (RAG) is a technique that enhances LLMs with external knowledge.",
            metadata={"source": "rag_paper.pdf", "page": 1},
        ),
        Document(
            page_content="LangGraph provides a framework for building stateful, multi-actor applications with LLMs using graph topology.",
            metadata={"source": "langgraph_docs.md"},
        ),
        Document(
            page_content="Qdrant is a vector database optimized for high-performance similarity search at scale.",
            metadata={"source": "qdrant_overview.html"},
        ),
    ]
