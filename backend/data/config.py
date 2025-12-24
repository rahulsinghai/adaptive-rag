from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorBackend(str, Enum):
    qdrant = "qdrant"
    faiss = "faiss"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., description="OpenAI API key")
    tavily_api_key: str = Field(..., description="Tavily search API key")
    mongodb_uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection URI")
    mongodb_db_name: str = Field(default="adaptive_rag", description="MongoDB database name")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str | None = Field(default=None, description="Qdrant API key (optional for local)")
    qdrant_collection: str = Field(default="documents", description="Qdrant collection name")
    faiss_index_path: str = Field(default="./faiss_index", description="Local path to persist FAISS index")
    langsmith_api_key: str | None = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="adaptive-rag-ops-lab", description="LangSmith project name")
    langsmith_tracing: bool = Field(default=True, description="Enable LangSmith tracing")
    feature_vector_backend: VectorBackend = Field(default=VectorBackend.qdrant, description="Vector DB backend")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    embedding_dimensions: int = Field(default=1536, description="Embedding vector dimensions")
    retrieval_top_k: int = Field(default=5, description="Top-k docs to retrieve")
    confidence_threshold: float = Field(default=0.7, description="Min confidence to skip web search")
    request_timeout: int = Field(default=30, description="Timeout seconds for external calls")
    log_level: str = Field(default="INFO", description="Logging level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
