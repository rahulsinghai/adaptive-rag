"""MongoDB / Motor async chat repository."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    messages: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatRepository:
    def __init__(self, settings: Any) -> None:
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(  # type: ignore[type-arg]
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        self._db: AsyncIOMotorDatabase = self._client[settings.mongodb_db_name]  # type: ignore[type-arg]
        self._sessions: AsyncIOMotorCollection = self._db["chat_sessions"]  # type: ignore[type-arg]

    async def create_session(self, metadata: dict[str, Any] | None = None) -> ChatSession:
        session = ChatSession(metadata=metadata or {})
        await self._sessions.insert_one(session.model_dump())
        logger.info("Created session %s", session.session_id)
        return session

    async def get_session(self, session_id: str) -> ChatSession | None:
        doc = await self._sessions.find_one({"session_id": session_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        return ChatSession(**doc)

    async def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        serialized = [m.model_dump() for m in messages]
        result = await self._sessions.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"$each": serialized}}},
        )
        if result.matched_count == 0:
            raise ValueError(f"Session {session_id} not found")
        logger.debug("Appended %d messages to session %s", len(messages), session_id)

    async def list_sessions(self, limit: int = 50) -> list[ChatSession]:
        cursor = self._sessions.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        sessions = []
        async for doc in cursor:
            sessions.append(ChatSession(**doc))
        return sessions

    async def close(self) -> None:
        self._client.close()


class InMemoryChatRepository:
    """In-memory fallback — useful for testing without MongoDB."""

    def __init__(self) -> None:
        self._store: dict[str, ChatSession] = {}

    async def create_session(self, metadata: dict[str, Any] | None = None) -> ChatSession:
        session = ChatSession(metadata=metadata or {})
        self._store[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> ChatSession | None:
        return self._store.get(session_id)

    async def append_messages(self, session_id: str, messages: list[ChatMessage]) -> None:
        session = self._store.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.messages.extend(messages)

    async def list_sessions(self, limit: int = 50) -> list[ChatSession]:
        return list(self._store.values())[:limit]

    async def close(self) -> None:
        pass
