"""FastAPI dependency injection — singleton services per process."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from backend.data.chat_repo import ChatRepository, InMemoryChatRepository
from backend.data.config import Settings, get_settings
from backend.data.vector_store import VectorStoreAdapter, build_vector_store

logger = logging.getLogger(__name__)

# Module-level singletons (initialised lazily)
_vector_store: VectorStoreAdapter | None = None
_chat_repo: ChatRepository | InMemoryChatRepository | None = None


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> VectorStoreAdapter:
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(settings)
    return _vector_store


def get_chat_repo(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatRepository | InMemoryChatRepository:
    global _chat_repo
    if _chat_repo is None:
        if settings.mongodb_uri.startswith("memory://"):
            logger.info("Using in-memory chat repository")
            _chat_repo = InMemoryChatRepository()
        else:
            logger.info("Using MongoDB chat repository")
            _chat_repo = ChatRepository(settings)
    return _chat_repo


SettingsDep = Annotated[Settings, Depends(get_settings)]
VectorStoreDep = Annotated[VectorStoreAdapter, Depends(get_vector_store)]
ChatRepoDep = Annotated[ChatRepository | InMemoryChatRepository, Depends(get_chat_repo)]
