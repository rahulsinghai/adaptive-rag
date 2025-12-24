"""FastAPI application factory with LangSmith tracing setup."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.data.config import get_settings

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    # Enable LangSmith tracing via env vars before any LangChain imports run
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled for project '%s'", settings.langsmith_project)

    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="Adaptive RAG Ops Lab",
        description="Production-style adaptive RAG system with LangGraph orchestration.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Adaptive RAG API starting up — vector backend: %s", settings.feature_vector_backend)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        from backend.api.dependencies import _chat_repo

        if _chat_repo is not None:
            await _chat_repo.close()
        logger.info("Adaptive RAG API shut down")

    return app


app = create_app()
