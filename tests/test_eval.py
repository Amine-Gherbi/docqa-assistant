"""Tests for the R5 evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docqa.eval import (
    ANSWERABLE_COUNT,
    EXIT_GATE,
    STAGES,
    Aggregate,
    GoldError,
    Rate,
    attribute_stage,
    build_gates,
    exit_code,
    load_gold,
    wilson_interval,
)


def _aggregate(**overrides: Rate) -> Aggregate:
    rates = {
        "retrieval_recall": Rate(20, 24),
        "answer_rate": Rate(18, 24),
        "unanswerable_refusal_rate": Rate(6, 6),
        "citation_precision": Rate(20, 20),
        "faithfulness": Rate(20, 20),
    }
    rates.update(overrides)
    return Aggregate(
        **rates,
        stage_counts={stage: 0 for stage in STAGES},
    )


def test_gold_set_has_the_declared_shape() -> None:
    questions = load_gold()
    assert len(questions) == 30
    assert sum(question.answerable for question in questions) == ANSWERABLE_COUNT
    unanswerable = [question for question in questions if not question.answerable]
    assert len(unanswerable) == 6
    assert all(question.gold is None for question in unanswerable)
    assert all(question.gold is not None for question in questions if question.answerable)


def test_gold_loader_rejects_a_reduced_undocumented_set(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    row = {
        "id": "only",
        "question": "q",
        "kind": "answerable",
        "gold": {"doc_id": "d", "page_no": 1, "chunk_index": 0, "quote": "x"},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(GoldError, match="expected 24 answerable"):
        load_gold(path)


def test_wilson_interval_bounds_and_empty_sample() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    low, high = wilson_interval(0, 24)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high < 0.15
    low, high = wilson_interval(24, 24)
    assert high == pytest.approx(1.0)
    assert low > 0.85
    low, high = wilson_interval(12, 24)
    assert low < 0.5 < high


def test_missed_gate_makes_the_harness_exit_non_zero() -> None:
    metrics = _aggregate(retrieval_recall=Rate(10, 24))
    gates = build_gates(metrics)
    assert [(gate.name, gate.passed) for gate in gates][0] == (
        "retrieval_recall_at_8",
        False,
    )
    assert exit_code(gates) == EXIT_GATE


def test_all_gates_pass_gives_exit_zero() -> None:
    gates = build_gates(_aggregate())
    assert all(gate.passed for gate in gates)
    assert exit_code(gates) == 0


def test_empty_denominator_never_passes_vacuously() -> None:
    metrics = _aggregate(citation_precision=Rate(0, 0), faithfulness=Rate(0, 0))
    gates = {gate.name: gate for gate in build_gates(metrics)}
    assert gates["citation_precision"].passed is False
    assert gates["faithfulness"].passed is False
    assert "vacuously" in gates["faithfulness"].detail


def test_attribute_stage_names_the_earliest_loss() -> None:
    def stage(**overrides: object) -> str | None:
        values: dict[str, object] = {
            "mode": "hybrid_rerank",
            "answerable": True,
            "in_context": True,
            "in_vector": True,
            "in_lexical": True,
            "in_fusion": True,
            "refused": False,
            "refused_for_schema": False,
            "has_claims": True,
        }
        values.update(overrides)
        return attribute_stage(**values)  # type: ignore[arg-type]

    assert stage() is None
    assert stage(mode="vector", in_context=False) == "retrieval"
    assert stage(in_context=False, in_vector=False, in_lexical=False) == "retrieval"
    assert stage(mode="hybrid", in_context=False) == "fusion"
    assert stage(in_context=False) == "rerank"
    assert stage(refused=True) == "generation"
    assert stage(refused=True, refused_for_schema=True) == "schema"
    assert stage(answerable=False, refused=True) is None
    assert stage(answerable=False, refused=False) == "generation"
