"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    file_path: str = Field(..., description="Absolute or relative path to PDF/Markdown/HTML file")

    model_config = {"json_schema_extra": {"example": {"file_path": "/data/my_paper.pdf"}}}


class IngestResponse(BaseModel):
    status: str
    file_path: str
    chunks_stored: int


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Existing session ID; omit to create new")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is adaptive RAG?",
                "session_id": None,
            }
        }
    }


class SourceItem(BaseModel):
    index: int
    type: str  # "local" | "web"
    source: str
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: str
    confidence: float
    sources: list[SourceItem]
    latency_ms: float | None
    token_usage: dict[str, int]


# ── Session history ───────────────────────────────────────────────────────────

class MessageOut(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageOut]


# ── Eval ─────────────────────────────────────────────────────────────────────

class EvalRunResponse(BaseModel):
    timestamp: str
    dataset: str
    num_examples: int
    avg_keyword_relevance: float
    pass_: bool = Field(alias="pass")

    model_config = {"populate_by_name": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    services: dict[str, Any] = Field(default_factory=dict)
