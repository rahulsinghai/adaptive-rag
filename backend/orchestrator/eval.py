"""LangSmith eval pipeline — creates a dataset and runs automated evals."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_DATASET_NAME = "adaptive-rag-eval-v1"

SEED_EXAMPLES = [
    {
        "question": "What is retrieval-augmented generation?",
        "expected_keywords": ["retrieval", "augmented", "generation", "LLM", "documents"],
    },
    {
        "question": "How does LangGraph handle conditional routing?",
        "expected_keywords": ["conditional", "edge", "graph", "node", "state"],
    },
    {
        "question": "What are the advantages of using Qdrant over FAISS?",
        "expected_keywords": ["Qdrant", "FAISS", "vector", "index", "scalability"],
    },
]


def _get_ls_client(settings: Any) -> Any:
    from langsmith import Client

    return Client(api_key=settings.langsmith_api_key)


async def ensure_eval_dataset(settings: Any) -> str:
    """Create or return the eval dataset in LangSmith."""
    client = _get_ls_client(settings)

    datasets = list(client.list_datasets(dataset_name=EVAL_DATASET_NAME))
    if datasets:
        ds_id = datasets[0].id
        logger.info("Using existing LangSmith dataset: %s (%s)", EVAL_DATASET_NAME, ds_id)
        return str(ds_id)

    dataset = client.create_dataset(EVAL_DATASET_NAME, description="Adaptive RAG eval set")
    for ex in SEED_EXAMPLES:
        client.create_example(
            inputs={"question": ex["question"]},
            outputs={"expected_keywords": ex["expected_keywords"]},
            dataset_id=dataset.id,
        )
    logger.info("Created LangSmith dataset %s with %d examples", EVAL_DATASET_NAME, len(SEED_EXAMPLES))
    return str(dataset.id)


def _keyword_relevance(run_outputs: dict[str, Any], example_outputs: dict[str, Any]) -> dict[str, Any]:
    answer: str = run_outputs.get("answer", "").lower()
    keywords: list[str] = example_outputs.get("expected_keywords", [])
    hits = sum(1 for kw in keywords if kw.lower() in answer)
    score = hits / len(keywords) if keywords else 0.0
    return {"score": score, "key": "keyword_relevance"}


async def run_eval(settings: Any, pipeline_fn: Any) -> dict[str, Any]:
    """
    Run the eval pipeline against the LangSmith dataset.
    Returns a summary dict and writes a report to eval_report.json / eval_report.md.
    """
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = _get_ls_client(settings)
    dataset_id = await ensure_eval_dataset(settings)

    async def _target(inputs: dict[str, Any]) -> dict[str, Any]:
        state = await pipeline_fn(question=inputs["question"], session_id="eval-session")
        return {
            "answer": state.answer,
            "route": state.route.value if state.route else None,
            "confidence": state.confidence,
        }

    # langsmith evaluate is sync; run inline
    results = evaluate(
        _target,  # type: ignore[arg-type]
        data=dataset_id,
        evaluators=[_keyword_relevance],
        experiment_prefix="adaptive-rag",
        client=client,
    )

    scores = [r.get("evaluation_results", {}).get("results", [{}])[0].get("score", 0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": EVAL_DATASET_NAME,
        "num_examples": len(SEED_EXAMPLES),
        "avg_keyword_relevance": round(avg_score, 4),
        "pass": avg_score >= 0.5,
    }

    _write_report(report)
    return report


def _write_report(report: dict[str, Any]) -> None:
    out_dir = Path("eval_reports")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"eval_report_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_path = out_dir / f"eval_report_{ts}.md"
    md_path.write_text(
        f"# Eval Report — {report['timestamp']}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Dataset | {report['dataset']} |\n"
        f"| Examples | {report['num_examples']} |\n"
        f"| Avg Keyword Relevance | {report['avg_keyword_relevance']:.4f} |\n"
        f"| Pass | {'✅' if report['pass'] else '❌'} |\n"
    )
    logger.info("Eval report written to %s and %s", json_path, md_path)
