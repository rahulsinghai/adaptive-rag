"""Eval pipeline test — verifies report artifact is emitted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.orchestrator.eval import _write_report
from backend.orchestrator.state import GraphState, RouteDecision


@pytest.mark.asyncio
async def test_eval_report_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval pipeline writes JSON + Markdown report artifacts."""
    monkeypatch.chdir(tmp_path)

    report = {
        "timestamp": "2026-03-13T00:00:00+00:00",
        "dataset": "adaptive-rag-eval-v1",
        "num_examples": 3,
        "avg_keyword_relevance": 0.72,
        "pass": True,
    }
    _write_report(report)

    report_dir = tmp_path / "eval_reports"
    json_files = list(report_dir.glob("*.json"))
    md_files = list(report_dir.glob("*.md"))

    assert len(json_files) == 1, "Expected 1 JSON report"
    assert len(md_files) == 1, "Expected 1 Markdown report"

    loaded = json.loads(json_files[0].read_text())
    assert loaded["avg_keyword_relevance"] == 0.72
    assert loaded["pass"] is True

    md_content = md_files[0].read_text()
    assert "Eval Report" in md_content
    assert "adaptive-rag-eval-v1" in md_content
    assert "✅" in md_content


@pytest.mark.asyncio
async def test_eval_pipeline_calls_langsmith(mock_settings: Any) -> None:
    """run_eval calls LangSmith ensure_eval_dataset and evaluate."""
    final_state = GraphState(
        question="test",
        session_id="eval-session",
        answer="test answer with retrieval and augmented generation LLM documents",
        route=RouteDecision.local_rag,
        confidence=0.9,
    )

    async def _mock_pipeline(question: str, session_id: str) -> GraphState:
        return final_state

    mock_client = MagicMock()
    mock_client.list_datasets = MagicMock(return_value=[MagicMock(id="ds-123")])
    mock_client.create_example = MagicMock()

    with (
        patch("backend.orchestrator.eval._get_ls_client", return_value=mock_client),
        patch(
            "backend.orchestrator.eval.evaluate",
            return_value=[{"evaluation_results": {"results": [{"score": 0.8}]}}],
        ),
        patch("backend.orchestrator.eval._write_report"),
    ):
        from backend.orchestrator.eval import run_eval

        mock_settings.langsmith_api_key = "ls-test"
        report = await run_eval(mock_settings, _mock_pipeline)

    assert "avg_keyword_relevance" in report
    assert "pass" in report


@pytest.mark.asyncio
async def test_ensure_eval_dataset_creates_if_missing(mock_settings: Any) -> None:
    """ensure_eval_dataset creates dataset when none exists."""
    mock_client = MagicMock()
    mock_client.list_datasets = MagicMock(return_value=[])
    mock_dataset = MagicMock()
    mock_dataset.id = "new-ds-456"
    mock_client.create_dataset = MagicMock(return_value=mock_dataset)
    mock_client.create_example = MagicMock()

    with patch("backend.orchestrator.eval._get_ls_client", return_value=mock_client):
        from backend.orchestrator.eval import ensure_eval_dataset

        mock_settings.langsmith_api_key = "ls-test"
        ds_id = await ensure_eval_dataset(mock_settings)

    assert ds_id == "new-ds-456"
    mock_client.create_dataset.assert_called_once()
    assert mock_client.create_example.call_count == 3  # 3 seed examples
