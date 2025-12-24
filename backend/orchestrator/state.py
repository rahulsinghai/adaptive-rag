"""Shared state schema for the LangGraph adaptive RAG pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteDecision(str, Enum):
    local_rag = "LOCAL_RAG"
    web_search = "WEB_SEARCH"
    hybrid = "HYBRID"


class GraphState(BaseModel):
    """Mutable state threaded through each graph node."""

    # Core I/O
    question: str
    session_id: str
    answer: str = ""

    # Routing
    route: RouteDecision | None = None
    confidence: float = 0.0

    # Retrieved context
    local_docs: list[Document] = Field(default_factory=list)
    web_results: list[dict[str, Any]] = Field(default_factory=list)

    # Final sources merged from both retrievers
    sources: list[dict[str, Any]] = Field(default_factory=list)

    # Observability
    latency_ms: float | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
