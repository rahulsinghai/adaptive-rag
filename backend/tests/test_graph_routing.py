"""Tests for LangGraph routing behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.orchestrator.state import GraphState, RouteDecision


class MockLLMResult:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 20}


def _make_llm(response: str) -> Any:
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MockLLMResult(response))
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=MockLLMResult(response))
    return chain


@pytest.fixture
def mock_vector_store(sample_docs: list[Document]) -> Any:
    vs = AsyncMock()
    vs.similarity_search = AsyncMock(return_value=sample_docs)
    return vs


@pytest.mark.asyncio
async def test_local_rag_route(mock_settings: Any, mock_vector_store: Any, chat_repo: Any) -> None:
    """When LLM classifies as LOCAL_RAG and confidence is high, no web search."""
    with (
        patch("backend.orchestrator.nodes.classify_query") as mock_classify,
        patch("backend.orchestrator.nodes.retrieve_local_context") as mock_retrieve,
        patch("backend.orchestrator.nodes.confidence_gate") as mock_conf,
        patch("backend.orchestrator.nodes.synthesize_answer") as mock_synth,
        patch("backend.orchestrator.nodes.persist_chat") as mock_persist,
    ):
        mock_classify.return_value = {"route": RouteDecision.local_rag}
        mock_retrieve.return_value = {"local_docs": [Document(page_content="RAG is great")]}
        mock_conf.return_value = {"confidence": 0.9}
        mock_synth.return_value = {
            "answer": "RAG augments LLMs with external knowledge.",
            "latency_ms": 100.0,
            "token_usage": {"input_tokens": 10, "output_tokens": 20},
            "sources": [{"index": 1, "type": "local", "source": "test.pdf", "snippet": "RAG is great"}],
        }
        mock_persist.return_value = {}

        from backend.orchestrator.graph import run_pipeline

        state = await run_pipeline(
            question="What is RAG?",
            session_id="test-session",
            settings=mock_settings,
            vector_store=mock_vector_store,
            chat_repo=chat_repo,
        )

    assert state.answer != ""
    assert state.route == RouteDecision.local_rag


@pytest.mark.asyncio
async def test_web_search_route(mock_settings: Any, mock_vector_store: Any, chat_repo: Any) -> None:
    """WEB_SEARCH route skips local retrieval entirely."""
    with (
        patch("backend.orchestrator.nodes.classify_query") as mock_classify,
        patch("backend.orchestrator.nodes.maybe_web_search") as mock_web,
        patch("backend.orchestrator.nodes.synthesize_answer") as mock_synth,
        patch("backend.orchestrator.nodes.persist_chat") as mock_persist,
    ):
        mock_classify.return_value = {"route": RouteDecision.web_search}
        mock_web.return_value = {
            "web_results": [{"url": "https://example.com", "content": "Latest AI news"}]
        }
        mock_synth.return_value = {
            "answer": "Latest AI developments include...",
            "latency_ms": 200.0,
            "token_usage": {"input_tokens": 15, "output_tokens": 30},
            "sources": [{"index": 1, "type": "web", "source": "https://example.com", "snippet": "Latest AI news"}],
        }
        mock_persist.return_value = {}

        from backend.orchestrator.graph import run_pipeline

        state = await run_pipeline(
            question="What happened in AI news today?",
            session_id="test-session",
            settings=mock_settings,
            vector_store=mock_vector_store,
            chat_repo=chat_repo,
        )

    assert state.answer != ""
    assert state.route == RouteDecision.web_search


@pytest.mark.asyncio
async def test_hybrid_route(mock_settings: Any, mock_vector_store: Any, chat_repo: Any) -> None:
    """HYBRID triggers both local retrieval and web search."""
    with (
        patch("backend.orchestrator.nodes.classify_query") as mock_classify,
        patch("backend.orchestrator.nodes.retrieve_local_context") as mock_retrieve,
        patch("backend.orchestrator.nodes.confidence_gate") as mock_conf,
        patch("backend.orchestrator.nodes.maybe_web_search") as mock_web,
        patch("backend.orchestrator.nodes.synthesize_answer") as mock_synth,
        patch("backend.orchestrator.nodes.persist_chat") as mock_persist,
    ):
        mock_classify.return_value = {"route": RouteDecision.hybrid}
        mock_retrieve.return_value = {"local_docs": [Document(page_content="local doc content")]}
        mock_conf.return_value = {"confidence": 0.6}
        mock_web.return_value = {"web_results": [{"url": "https://example.com", "content": "web content"}]}
        mock_synth.return_value = {
            "answer": "Combined answer from local and web.",
            "latency_ms": 350.0,
            "token_usage": {"input_tokens": 25, "output_tokens": 45},
            "sources": [],
        }
        mock_persist.return_value = {}

        from backend.orchestrator.graph import run_pipeline

        state = await run_pipeline(
            question="What are the latest advances in RAG?",
            session_id="test-session",
            settings=mock_settings,
            vector_store=mock_vector_store,
            chat_repo=chat_repo,
        )

    assert state.answer != ""
    assert state.route == RouteDecision.hybrid


@pytest.mark.asyncio
async def test_low_confidence_triggers_web_search(
    mock_settings: Any, mock_vector_store: Any, chat_repo: Any
) -> None:
    """LOCAL_RAG with confidence below threshold should fall back to web search."""
    with (
        patch("backend.orchestrator.nodes.classify_query") as mock_classify,
        patch("backend.orchestrator.nodes.retrieve_local_context") as mock_retrieve,
        patch("backend.orchestrator.nodes.confidence_gate") as mock_conf,
        patch("backend.orchestrator.nodes.maybe_web_search") as mock_web,
        patch("backend.orchestrator.nodes.synthesize_answer") as mock_synth,
        patch("backend.orchestrator.nodes.persist_chat") as mock_persist,
    ):
        mock_classify.return_value = {"route": RouteDecision.local_rag}
        mock_retrieve.return_value = {"local_docs": [Document(page_content="Unrelated content")]}
        mock_conf.return_value = {"confidence": 0.3}  # below threshold of 0.7
        mock_web.return_value = {"web_results": [{"url": "https://web.com", "content": "Better content"}]}
        mock_synth.return_value = {
            "answer": "Answer augmented by web.",
            "latency_ms": 280.0,
            "token_usage": {},
            "sources": [],
        }
        mock_persist.return_value = {}

        from backend.orchestrator.graph import run_pipeline

        state = await run_pipeline(
            question="Very specific question not in local docs",
            session_id="test-session",
            settings=mock_settings,
            vector_store=mock_vector_store,
            chat_repo=chat_repo,
        )

    assert state.answer != ""
    mock_web.assert_called_once()
