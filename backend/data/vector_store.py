"""Vector store adapter — Qdrant default, FAISS fallback via FEATURE_VECTOR_BACKEND env."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStoreAdapter(ABC):
    @abstractmethod
    async def upsert(self, documents: list[Document]) -> int:
        """Embed and upsert documents. Returns count inserted."""

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Return top-k documents by similarity."""

    @abstractmethod
    async def as_retriever(self, k: int = 5) -> Any:
        """Return a LangChain retriever interface."""


class QdrantAdapter(VectorStoreAdapter):
    def __init__(self, settings: Any) -> None:
        from langchain_openai import OpenAIEmbeddings
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._client: AsyncQdrantClient | None = None
        self._collection = settings.qdrant_collection

    async def _ensure_client(self) -> "AsyncQdrantClient":  # type: ignore[name-defined]
        if self._client is None:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = AsyncQdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key,
                timeout=self._settings.request_timeout,
            )
            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]
            if self._collection not in existing:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._settings.embedding_dimensions,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s'", self._collection)
        return self._client

    async def upsert(self, documents: list[Document]) -> int:
        from langchain_qdrant import QdrantVectorStore

        client = await self._ensure_client()
        store = QdrantVectorStore(
            client=client,  # type: ignore[arg-type]
            collection_name=self._collection,
            embedding=self._embeddings,
        )
        await store.aadd_documents(documents)
        logger.info("Upserted %d documents to Qdrant", len(documents))
        return len(documents)

    async def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        from langchain_qdrant import QdrantVectorStore

        client = await self._ensure_client()
        store = QdrantVectorStore(
            client=client,  # type: ignore[arg-type]
            collection_name=self._collection,
            embedding=self._embeddings,
        )
        return await store.asimilarity_search(query, k=k)

    async def as_retriever(self, k: int = 5) -> Any:
        from langchain_qdrant import QdrantVectorStore

        client = await self._ensure_client()
        store = QdrantVectorStore(
            client=client,  # type: ignore[arg-type]
            collection_name=self._collection,
            embedding=self._embeddings,
        )
        return store.as_retriever(search_kwargs={"k": k})


class FAISSAdapter(VectorStoreAdapter):
    """FAISS-backed local vector store — fallback when Qdrant unavailable."""

    def __init__(self, settings: Any) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._settings = settings
        self._embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._index_path = Path(settings.faiss_index_path)
        self._store: Any = None

    def _load_or_create(self) -> Any:
        if self._store is not None:
            return self._store
        try:
            from langchain_community.vectorstores import FAISS

            if (self._index_path / "index.faiss").exists():
                self._store = FAISS.load_local(
                    str(self._index_path),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("Loaded existing FAISS index from %s", self._index_path)
            else:
                # Initialise with a placeholder to avoid empty-index errors;
                # real docs will be upserted before first query.
                self._store = None
        except Exception as exc:
            logger.warning("FAISS load failed: %s", exc)
            self._store = None
        return self._store

    async def upsert(self, documents: list[Document]) -> int:
        from langchain_community.vectorstores import FAISS

        store = self._load_or_create()
        if store is None:
            self._store = await FAISS.afrom_documents(documents, self._embeddings)
        else:
            await self._store.aadd_documents(documents)
        self._index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self._index_path))
        logger.info("Upserted %d documents to FAISS", len(documents))
        return len(documents)

    async def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        store = self._load_or_create()
        if store is None:
            return []
        return await store.asimilarity_search(query, k=k)

    async def as_retriever(self, k: int = 5) -> Any:
        store = self._load_or_create()
        if store is None:
            return None
        return store.as_retriever(search_kwargs={"k": k})


def build_vector_store(settings: Any) -> VectorStoreAdapter:
    from backend.data.config import VectorBackend

    if settings.feature_vector_backend == VectorBackend.faiss:
        logger.info("Using FAISS vector backend")
        return FAISSAdapter(settings)
    logger.info("Using Qdrant vector backend")
    return QdrantAdapter(settings)
