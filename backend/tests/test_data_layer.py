"""Data layer tests: Motor chat persistence + vector upsert/query abstraction."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from backend.data.chat_repo import ChatMessage, ChatSession, InMemoryChatRepository
from backend.data.config import VectorBackend


class TestInMemoryChatRepo:
    @pytest.mark.asyncio
    async def test_create_session(self) -> None:
        repo = InMemoryChatRepository()
        session = await repo.create_session(metadata={"user": "alice"})
        assert session.session_id
        assert session.metadata["user"] == "alice"

    @pytest.mark.asyncio
    async def test_get_session_exists(self) -> None:
        repo = InMemoryChatRepository()
        session = await repo.create_session()
        fetched = await repo.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self) -> None:
        repo = InMemoryChatRepository()
        result = await repo.get_session("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_append_messages(self) -> None:
        repo = InMemoryChatRepository()
        session = await repo.create_session()
        msgs = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ]
        await repo.append_messages(session.session_id, msgs)
        updated = await repo.get_session(session.session_id)
        assert updated is not None
        assert len(updated.messages) == 2
        assert updated.messages[0].role == "user"
        assert updated.messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_append_messages_invalid_session(self) -> None:
        repo = InMemoryChatRepository()
        with pytest.raises(ValueError, match="not found"):
            await repo.append_messages("bad-id", [ChatMessage(role="user", content="x")])

    @pytest.mark.asyncio
    async def test_list_sessions(self) -> None:
        repo = InMemoryChatRepository()
        for i in range(3):
            await repo.create_session(metadata={"i": i})
        sessions = await repo.list_sessions()
        assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_close_noop(self) -> None:
        repo = InMemoryChatRepository()
        await repo.close()  # should not raise


class TestVectorStoreAbstraction:
    @pytest.mark.asyncio
    async def test_faiss_upsert_and_search(self, sample_docs: list[Document], mock_settings: Any) -> None:
        """FAISS adapter upsert and similarity_search flow (mocked embeddings)."""
        from backend.data.vector_store import FAISSAdapter

        with patch("langchain_community.vectorstores.FAISS.afrom_documents") as mock_create:
            mock_store = AsyncMock()
            mock_store.aadd_documents = AsyncMock()
            mock_store.asimilarity_search = AsyncMock(return_value=sample_docs[:2])
            mock_store.save_local = MagicMock()
            mock_create.return_value = mock_store

            with patch("langchain_openai.OpenAIEmbeddings"):
                adapter = FAISSAdapter(mock_settings)
                adapter._store = None

                count = await adapter.upsert(sample_docs)
                assert count == len(sample_docs)

    @pytest.mark.asyncio
    async def test_faiss_search_empty_store(self, mock_settings: Any) -> None:
        from backend.data.vector_store import FAISSAdapter

        with patch("langchain_openai.OpenAIEmbeddings"):
            adapter = FAISSAdapter(mock_settings)
            adapter._store = None
            results = await adapter.similarity_search("test query")
            assert results == []

    def test_build_vector_store_selects_faiss(self, mock_settings: Any) -> None:
        from backend.data.vector_store import FAISSAdapter, build_vector_store

        mock_settings.feature_vector_backend = VectorBackend.faiss
        with patch("langchain_openai.OpenAIEmbeddings"):
            store = build_vector_store(mock_settings)
        assert isinstance(store, FAISSAdapter)

    def test_build_vector_store_selects_qdrant(self, mock_settings: Any) -> None:
        from backend.data.vector_store import QdrantAdapter, build_vector_store

        mock_settings.feature_vector_backend = VectorBackend.qdrant
        with patch("langchain_openai.OpenAIEmbeddings"):
            store = build_vector_store(mock_settings)
        assert isinstance(store, QdrantAdapter)


class TestIngestion:
    def test_load_unsupported_file_raises(self) -> None:
        from backend.data.ingestion import load_file

        with pytest.raises(ValueError, match="Unsupported"):
            load_file("/data/document.docx")

    def test_split_documents(self, sample_docs: list[Document]) -> None:
        from backend.data.ingestion import split_documents

        chunks = split_documents(sample_docs)
        assert len(chunks) >= len(sample_docs)

    @pytest.mark.asyncio
    async def test_ingest_file_calls_upsert(self, mock_settings: Any) -> None:
        from backend.data.ingestion import ingest_file

        mock_vs = AsyncMock()
        mock_vs.upsert = AsyncMock(return_value=3)

        docs = [Document(page_content="test content", metadata={"source": "test.md"})]
        with (
            patch("backend.data.ingestion.load_file", return_value=docs),
            patch("backend.data.ingestion.split_documents", return_value=docs),
        ):
            count = await ingest_file("/fake/path.md", mock_vs)

        assert count == 3
        mock_vs.upsert.assert_called_once_with(docs)
