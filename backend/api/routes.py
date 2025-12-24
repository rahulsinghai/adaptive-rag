"""FastAPI route handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.api.dependencies import ChatRepoDep, SettingsDep, VectorStoreDep
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvalRunResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MessageOut,
    SessionHistoryResponse,
    SourceItem,
)
from backend.data.ingestion import ingest_file
from backend.orchestrator.graph import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        services={
            "vector_backend": settings.feature_vector_backend.value,
            "langsmith_tracing": settings.langsmith_tracing,
        }
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest(
    body: IngestRequest,
    vector_store: VectorStoreDep,
) -> IngestResponse:
    try:
        count = await ingest_file(body.file_path, vector_store)
        return IngestResponse(status="ok", file_path=body.file_path, chunks_stored=count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(
    body: ChatRequest,
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    chat_repo: ChatRepoDep,
) -> ChatResponse:
    # Create session if not provided
    session_id = body.session_id
    if session_id is None:
        session = await chat_repo.create_session()
        session_id = session.session_id

    try:
        state = await run_pipeline(
            question=body.question,
            session_id=session_id,
            settings=settings,
            vector_store=vector_store,
            chat_repo=chat_repo,
        )
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [SourceItem(**s) for s in state.sources]
    return ChatResponse(
        session_id=session_id,
        answer=state.answer,
        route=state.route.value if state.route else "UNKNOWN",
        confidence=state.confidence,
        sources=sources,
        latency_ms=state.latency_ms,
        token_usage=state.token_usage,
    )


@router.get("/chat/{session_id}", response_model=SessionHistoryResponse, tags=["chat"])
async def get_chat_history(
    session_id: str,
    chat_repo: ChatRepoDep,
) -> SessionHistoryResponse:
    session = await chat_repo.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    messages = [
        MessageOut(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp.isoformat(),
        )
        for m in session.messages
    ]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@router.post("/eval/run", response_model=EvalRunResponse, tags=["eval"])
async def eval_run(
    settings: SettingsDep,
    vector_store: VectorStoreDep,
    chat_repo: ChatRepoDep,
) -> EvalRunResponse:
    from backend.orchestrator.eval import run_eval

    async def _pipeline(question: str, session_id: str) -> Any:
        return await run_pipeline(
            question=question,
            session_id=session_id,
            settings=settings,
            vector_store=vector_store,
            chat_repo=chat_repo,
        )

    try:
        report = await run_eval(settings, _pipeline)
    except Exception as exc:
        logger.exception("Eval run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return EvalRunResponse(
        timestamp=report["timestamp"],
        dataset=report["dataset"],
        num_examples=report["num_examples"],
        avg_keyword_relevance=report["avg_keyword_relevance"],
        **{"pass": report["pass"]},
    )
