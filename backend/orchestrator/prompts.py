"""Prompts used across the orchestrator graph."""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

CLASSIFY_PROMPT = PromptTemplate.from_template(
    """You are a query classifier for an adaptive RAG system.

Given the user question below, determine the best retrieval strategy:
- "LOCAL_RAG"   — the answer is likely contained in the ingested document base
- "WEB_SEARCH"  — the question requires up-to-date or external information
- "HYBRID"      — both local docs and a live web search would improve the answer

Respond with ONLY one of: LOCAL_RAG, WEB_SEARCH, HYBRID

Question: {question}
Strategy:"""
)

CONFIDENCE_PROMPT = PromptTemplate.from_template(
    """You are evaluating whether retrieved context is sufficient to answer a question.

Question: {question}

Retrieved context:
{context}

On a scale from 0.0 to 1.0, how confident are you that the above context contains enough information
to answer the question accurately? Respond with ONLY a float between 0.0 and 1.0.

Confidence:"""
)

SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Answer the question based on the provided context. "
            "If you use information from a source, reference it with [Source N]. "
            "Be concise, accurate, and well-structured.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:",
        ),
    ]
)
