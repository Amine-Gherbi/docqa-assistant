"""Fit the refusal calibration on the human gold questions.

Why this differs from ``scripts/fit_calibration.py``: the provisional fitter
generated queries from corpus sentences, so its scores described passage-like
text rather than real questions. This fitter uses the 30 hand-authored gold
questions and calibrates the statistic the refusal decision actually consumes:
the **top** reranked score for the question. Each question is one observation,
labelled ``1`` when it is answerable and ``0`` when it is not.

The threshold maximises balanced accuracy on the fit split. The script reports
both held-out directions (odd ids fitted, even ids evaluated; and the reverse)
and then refits on all 30 for the production artifact. The final coefficients
and threshold are therefore **in-sample** with respect to ``eval/gold.jsonl``;
the artifact says so and ``docs/EVAL.md`` repeats the caveat.

Run (local only; no model endpoint is contacted):

    uv run python scripts/fit_calibration_gold.py
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from docqa.answer import calibration_fit_hash
from docqa.calibration_fit import (
    Pair,
    choose_threshold,
    classification_rates,
    fit_platt,
    sigmoid,
)
from docqa.config import PROJECT_ROOT, Settings, get_settings
from docqa.db import make_engine, make_session_factory, session_scope
from docqa.eval import DEFAULT_GOLD_PATH, load_gold, resolve_gold
from docqa.ingest.embedder import build_embedder
from docqa.rerank import build_reranker
from docqa.retrieval import PgCandidateSource, retrieve_with_trace

FIT_SET_PATH = PROJECT_ROOT / "eval" / "calibration_fit_set.json"
CALIBRATION_PATH = PROJECT_ROOT / "eval" / "calibration.json"
_ID_PARITY = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class Observation:
    """One gold question, its top reranked score, and its label."""

    id: str
    question: str
    answerable: bool
    top_score: float
    gold_chunk_id: int | None

    @property
    def label(self) -> int:
        """``1`` for an answerable question, ``0`` otherwise."""
        return 1 if self.answerable else 0

    @property
    def parity(self) -> int:
        """The numeric suffix's parity; odd ids are ``1``, even ids are ``0``."""
        match = _ID_PARITY.search(self.id)
        if match is None:
            raise ValueError(f"question id {self.id!r} has no numeric suffix")
        return int(match.group(1)) % 2


async def collect_observations(settings: Settings) -> tuple[list[Observation], str]:
    """Retrieve every gold question and record its top reranked score."""
    questions = load_gold(DEFAULT_GOLD_PATH)
    embedder_choice = build_embedder(settings)
    reranker_choice = build_reranker(mode="auto")
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    observations: list[Observation] = []
    try:
        async with session_scope(factory) as session:
            gold = await resolve_gold(questions, session)
            source = PgCandidateSource(session)
            for question in questions:
                result = await retrieve_with_trace(
                    question.question,
                    8,
                    embedder=embedder_choice.embedder,
                    source=source,
                    reranker=reranker_choice.reranker,
                )
                if not result.chunks:
                    raise RuntimeError(
                        f"{question.id}: retrieval returned no context; the "
                        "refusal calibration cannot be fitted on a missing score"
                    )
                top_score = result.chunks[0].rerank_score
                if top_score is None:
                    raise RuntimeError(
                        f"{question.id}: the production path returned no rerank score"
                    )
                resolved = gold.get(question.id)
                observations.append(
                    Observation(
                        id=question.id,
                        question=question.question,
                        answerable=question.answerable,
                        top_score=top_score,
                        gold_chunk_id=resolved.chunk_id if resolved is not None else None,
                    )
                )
    finally:
        await engine.dispose()
    return observations, reranker_choice.reranker.name


def question_pairs(observations: list[Observation]) -> list[Pair]:
    """One calibration pair per question."""
    return [
        Pair(
            query=observation.question,
            chunk_id=observation.gold_chunk_id,
            label=observation.label,
            score=observation.top_score,
        )
        for observation in observations
    ]


def _question_decisions(
    observations: list[Observation], a: float, b: float, threshold: float
) -> dict[str, object]:
    """Count what the threshold would do at the question level."""
    admitted = sum(
        1
        for observation in observations
        if observation.answerable and sigmoid(a * observation.top_score + b) >= threshold
    )
    answerable = sum(1 for observation in observations if observation.answerable)
    refused = sum(
        1
        for observation in observations
        if not observation.answerable
        and sigmoid(a * observation.top_score + b) < threshold
    )
    unanswerable = sum(1 for observation in observations if not observation.answerable)
    return {
        "answerable_admitted": admitted,
        "answerable_total": answerable,
        "unanswerable_refused": refused,
        "unanswerable_total": unanswerable,
    }


def _fit_and_evaluate(
    train: list[Observation], test: list[Observation]
) -> dict[str, object]:
    """Fit on ``train``, pick the threshold there, then measure ``test``."""
    train_pairs = question_pairs(train)
    test_pairs = question_pairs(test)
    a, b = fit_platt(train_pairs)
    threshold, train_accuracy = choose_threshold(train_pairs, a, b)
    test_tpr, test_tnr = classification_rates(test_pairs, a, b, threshold)
    return {
        "fit_question_ids": [observation.id for observation in train],
        "evaluated_question_ids": [observation.id for observation in test],
        "fit_size": len(train_pairs),
        "fit_positives": sum(1 for pair in train_pairs if pair.label == 1),
        "fit_negatives": sum(1 for pair in train_pairs if pair.label == 0),
        "a": a,
        "b": b,
        "threshold": threshold,
        "fit_balanced_accuracy": train_accuracy,
        "evaluated_true_positive_rate": test_tpr,
        "evaluated_true_negative_rate": test_tnr,
        "evaluated_balanced_accuracy": (test_tpr + test_tnr) / 2.0,
        "evaluated_question_decisions": _question_decisions(test, a, b, threshold),
    }


def main() -> int:
    """Fit, write the fit-set artifact and ``eval/calibration.json``, print the numbers."""
    settings = get_settings()
    observations, reranker_name = asyncio.run(collect_observations(settings))
    if sum(1 for observation in observations if observation.answerable) != 24:
        raise RuntimeError("expected 24 answerable gold questions")
    if sum(1 for observation in observations if not observation.answerable) != 6:
        raise RuntimeError("expected 6 unanswerable gold questions")

    odd = [observation for observation in observations if observation.parity == 1]
    even = [observation for observation in observations if observation.parity == 0]
    held_out = {
        "fit_odd_eval_even": _fit_and_evaluate(odd, even),
        "fit_even_eval_odd": _fit_and_evaluate(even, odd),
    }

    all_pairs = question_pairs(observations)
    a, b = fit_platt(all_pairs)
    threshold, in_sample_accuracy = choose_threshold(all_pairs, a, b)
    in_sample_tpr, in_sample_tnr = classification_rates(
        all_pairs, a, b, threshold
    )

    FIT_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fit_rows = [
        {
            "question_id": observation.id,
            "query": observation.question,
            "chunk_id": observation.gold_chunk_id,
            "label": observation.label,
            "rerank_score": observation.top_score,
            "split": "odd" if observation.parity == 1 else "even",
        }
        for observation in observations
    ]
    FIT_SET_PATH.write_text(
        json.dumps(fit_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fit_hash = calibration_fit_hash(FIT_SET_PATH)

    calibration = {
        "a": a,
        "b": b,
        "threshold": threshold,
        "fit_set_sha256": fit_hash,
        "fit_size": len(all_pairs),
        "positives": sum(1 for pair in all_pairs if pair.label == 1),
        "negatives": sum(1 for pair in all_pairs if pair.label == 0),
        "mean_balanced_accuracy": in_sample_accuracy,
        "reranker": reranker_name,
        "provenance": (
            "gold-human-v1: question-level Platt scaling over the top reranked score "
            "of the 30 hand-authored gold questions (24 answerable, 6 unanswerable); "
            "the threshold maximises balanced accuracy. Coefficients and threshold "
            "are refitted on all 30, so the reported in-sample balanced accuracy is "
            "optimistic; held_out records the id-parity split measured out of sample."
        ),
        "fitted_at": datetime.now(UTC).date().isoformat(),
        "split_scheme": "id-parity (odd/even numeric suffix)",
        "held_out": held_out,
        "in_sample": {
            "a": a,
            "b": b,
            "threshold": threshold,
            "true_positive_rate": in_sample_tpr,
            "true_negative_rate": in_sample_tnr,
            "balanced_accuracy": in_sample_accuracy,
            "question_decisions": _question_decisions(
                observations, a, b, threshold
            ),
        },
    }
    CALIBRATION_PATH.write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"reranker: {reranker_name}")
    for label, report in held_out.items():
        print(
            f"{label}: threshold={report['threshold']:.6f} "
            f"fit_BA={report['fit_balanced_accuracy']:.4f} "
            f"held_out_BA={report['evaluated_balanced_accuracy']:.4f} "
            f"held_out_TPR={report['evaluated_true_positive_rate']:.3f} "
            f"held_out_TNR={report['evaluated_true_negative_rate']:.3f} "
            f"decisions={report['evaluated_question_decisions']}"
        )
    print(
        f"in-sample: a={a:.6f} b={b:.6f} threshold={threshold:.6f} "
        f"BA={in_sample_accuracy:.4f} TPR={in_sample_tpr:.3f} "
        f"TNR={in_sample_tnr:.3f} "
        f"decisions={calibration['in_sample']['question_decisions']}"
    )
    print(f"fit_set_sha256={fit_hash}")
    print(f"wrote {CALIBRATION_PATH.relative_to(PROJECT_ROOT)} and "
          f"{FIT_SET_PATH.relative_to(PROJECT_ROOT)}")
    print("caveat: the production threshold is fitted on all 30 gold questions and "
          "is therefore in-sample; the held_out blocks are the out-of-sample numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
