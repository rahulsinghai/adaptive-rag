"""LangGraph adaptive RAG pipeline with conditional routing edges."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from backend.orchestrator.nodes import (
    classify_query,
    confidence_gate,
    maybe_web_search,
    persist_chat,
    retrieve_local_context,
    synthesize_answer,
)
from backend.orchestrator.state import GraphState, RouteDecision

logger = logging.getLogger(__name__)


def _route_after_classify(state: GraphState) -> Literal["retrieve_local", "web_search_only"]:
    """After classify_query: WEB_SEARCH → skip local retrieval."""
    if state.route == RouteDecision.web_search:
        return "web_search_only"
    return "retrieve_local"


def _route_after_confidence(
    state: GraphState, settings: Any
) -> Literal["synthesize", "web_search_augment"]:
    """After confidence_gate: low confidence or HYBRID → augment with web."""
    if state.route == RouteDecision.hybrid or state.confidence < settings.confidence_threshold:
        return "web_search_augment"
    return "synthesize"


def build_graph(
    settings: Any,
    vector_store: Any,
    chat_repo: Any,
) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Routing logic:
      classify_query
          ├─ WEB_SEARCH  ──────────────────────────────► maybe_web_search ──► synthesize_answer
          └─ LOCAL_RAG / HYBRID ──► retrieve_local_context
                                        └─► confidence_gate
                                                ├─ HIGH confidence (LOCAL_RAG) ──► synthesize_answer
                                                └─ LOW confidence / HYBRID  ──► maybe_web_search ──► synthesize_answer
      synthesize_answer ──► persist_chat ──► END
    """
    graph = StateGraph(GraphState)

    # ── Node wrappers (bind dependencies) ───────────────────────────────────
    async def _classify(state: GraphState) -> dict[str, Any]:
        return await classify_query(state, settings)

    async def _retrieve(state: GraphState) -> dict[str, Any]:
        return await retrieve_local_context(state, vector_store, settings)

    async def _confidence(state: GraphState) -> dict[str, Any]:
        return await confidence_gate(state, settings)

    async def _web_search(state: GraphState) -> dict[str, Any]:
        return await maybe_web_search(state, settings)

    async def _synthesize(state: GraphState) -> dict[str, Any]:
        return await synthesize_answer(state, settings)

    async def _persist(state: GraphState) -> dict[str, Any]:
        return await persist_chat(state, chat_repo)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("classify_query", _classify)
    graph.add_node("retrieve_local", _retrieve)
    graph.add_node("confidence_gate", _confidence)
    graph.add_node("maybe_web_search", _web_search)
    graph.add_node("synthesize_answer", _synthesize)
    graph.add_node("persist_chat", _persist)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("classify_query")

    # ── Conditional edge: post-classify ─────────────────────────────────────
    graph.add_conditional_edges(
        "classify_query",
        _route_after_classify,
        {
            "retrieve_local": "retrieve_local",
            "web_search_only": "maybe_web_search",
        },
    )

    # ── Local retrieval → confidence gate ────────────────────────────────────
    graph.add_edge("retrieve_local", "confidence_gate")

    # ── Conditional edge: post-confidence ────────────────────────────────────
    graph.add_conditional_edges(
        "confidence_gate",
        lambda s: _route_after_confidence(s, settings),
        {
            "synthesize": "synthesize_answer",
            "web_search_augment": "maybe_web_search",
        },
    )

    # ── Web search feeds into synthesis ─────────────────────────────────────
    graph.add_edge("maybe_web_search", "synthesize_answer")

    # ── Synthesis → persist → done ───────────────────────────────────────────
    graph.add_edge("synthesize_answer", "persist_chat")
    graph.add_edge("persist_chat", END)

    return graph.compile()


async def run_pipeline(
    question: str,
    session_id: str,
    settings: Any,
    vector_store: Any,
    chat_repo: Any,
) -> GraphState:
    """Entry point: runs the compiled graph and returns final state."""
    pipeline = build_graph(settings, vector_store, chat_repo)
    initial = GraphState(question=question, session_id=session_id)
    result = await pipeline.ainvoke(initial)
    return GraphState(**result)
