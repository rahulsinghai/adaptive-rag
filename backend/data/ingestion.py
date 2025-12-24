"""Document ingestion pipeline using LangChain Community loaders/splitters."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.data.vector_store import VectorStoreAdapter

logger = logging.getLogger(__name__)

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    add_start_index=True,
)

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_file(file_path: str) -> list[Document]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(_LOADER_MAP)}")
    loader = loader_cls(str(path))
    docs = loader.load()
    logger.info("Loaded %d pages/sections from %s", len(docs), path.name)
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    chunks = _SPLITTER.split_documents(documents)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


async def ingest_file(file_path: str, vector_store: VectorStoreAdapter) -> int:
    """Load → split → upsert. Returns number of chunks stored."""
    raw = load_file(file_path)
    chunks = split_documents(raw)
    count = await vector_store.upsert(chunks)
    return count
