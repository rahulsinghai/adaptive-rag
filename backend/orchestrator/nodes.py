"""All LangGraph node implementations."""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable

from backend.orchestrator.prompts import CLASSIFY_PROMPT, CONFIDENCE_PROMPT, SYNTHESIZE_PROMPT
from backend.orchestrator.state import GraphState, RouteDecision

logger = logging.getLogger(__name__)


def _build_llm(settings: Any) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        request_timeout=settings.request_timeout,
        max_retries=3,
    )


def _docs_to_context(docs: list[Document], web: list[dict[str, Any]] | None = None) -> str:
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "local")
        parts.append(f"[Source {i}] {src}\n{doc.page_content}")
    offset = len(docs)
    for j, result in enumerate(web or [], offset + 1):
        parts.append(f"[Source {j}] {result.get('url', 'web')}\n{result.get('content', '')}")
    return "\n\n---\n\n".join(parts)


@traceable(name="classify_query")
async def classify_query(state: GraphState, settings: Any) -> dict[str, Any]:
    """Classify the query into a routing strategy."""
    llm = _build_llm(settings)
    chain = CLASSIFY_PROMPT | llm
    result = await chain.ainvoke({"question": state.question})
    raw = result.content.strip().upper()
    try:
        route = RouteDecision(raw)
    except ValueError:
        logger.warning("Unrecognised route '%s', defaulting to HYBRID", raw)
        route = RouteDecision.hybrid
    logger.info("Route decision: %s", route)
    return {"route": route}


@traceable(name="retrieve_local_context")
async def retrieve_local_context(state: GraphState, vector_store: Any, settings: Any) -> dict[str, Any]:
    """Retrieve top-k documents from the vector store."""
    try:
        docs = await vector_store.similarity_search(state.question, k=settings.retrieval_top_k)
        logger.info("Retrieved %d local docs", len(docs))
        return {"local_docs": docs}
    except Exception as exc:
        logger.error("Local retrieval failed: %s", exc)
        return {"local_docs": [], "error": str(exc)}


@traceable(name="confidence_gate")
async def confidence_gate(state: GraphState, settings: Any) -> dict[str, Any]:
    """Score confidence of local context; may trigger web search."""
    if not state.local_docs:
        return {"confidence": 0.0}
    llm = _build_llm(settings)
    chain = CONFIDENCE_PROMPT | llm
    context = _docs_to_context(state.local_docs)
    result = await chain.ainvoke({"question": state.question, "context": context})
    try:
        confidence = float(result.content.strip())
        confidence = max(0.0, min(1.0, confidence))
    except ValueError:
        confidence = 0.5
    logger.info("Confidence score: %.2f (threshold: %.2f)", confidence, settings.confidence_threshold)
    return {"confidence": confidence}


@traceable(name="maybe_web_search")
async def maybe_web_search(state: GraphState, settings: Any) -> dict[str, Any]:
    """Run Tavily web search if route calls for it."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        response = client.search(
            query=state.question,
            max_results=5,
            search_depth="advanced",
        )
        results = response.get("results", [])
        logger.info("Web search returned %d results", len(results))
        return {"web_results": results}
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return {"web_results": [], "error": str(exc)}


@traceable(name="synthesize_answer")
async def synthesize_answer(state: GraphState, settings: Any) -> dict[str, Any]:
    """Synthesize final answer from available context using OpenAI."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.2,
        request_timeout=settings.request_timeout,
        max_retries=3,
    )
    chain = SYNTHESIZE_PROMPT | llm
    context = _docs_to_context(state.local_docs, state.web_results)

    start = time.perf_counter()
    result = await chain.ainvoke({"question": state.question, "context": context})
    latency_ms = (time.perf_counter() - start) * 1000

    usage: dict[str, int] = {}
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        usage = {
            "input_tokens": result.usage_metadata.get("input_tokens", 0),
            "output_tokens": result.usage_metadata.get("output_tokens", 0),
        }

    # Build sources list
    sources: list[dict[str, Any]] = []
    for i, doc in enumerate(state.local_docs, 1):
        sources.append(
            {
                "index": i,
                "type": "local",
                "source": doc.metadata.get("source", "local"),
                "snippet": doc.page_content[:200],
            }
        )
    offset = len(state.local_docs)
    for j, r in enumerate(state.web_results, offset + 1):
        sources.append(
            {
                "index": j,
                "type": "web",
                "source": r.get("url", ""),
                "snippet": r.get("content", "")[:200],
            }
        )

    logger.info("Synthesized answer in %.0f ms, tokens=%s", latency_ms, usage)
    return {
        "answer": result.content,
        "latency_ms": latency_ms,
        "token_usage": usage,
        "sources": sources,
    }


@traceable(name="persist_chat")
async def persist_chat(state: GraphState, chat_repo: Any) -> dict[str, Any]:
    """Persist user question + assistant answer to MongoDB."""
    from backend.data.chat_repo import ChatMessage

    messages = [
        ChatMessage(role="user", content=state.question),
        ChatMessage(role="assistant", content=state.answer),
    ]
    try:
        await chat_repo.append_messages(state.session_id, messages)
        logger.info("Persisted chat to session %s", state.session_id)
    except Exception as exc:
        logger.error("Chat persist failed: %s", exc)
    return {}
