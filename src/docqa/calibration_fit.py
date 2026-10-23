"""Platt scaling and threshold selection for the refusal calibration.

The fit is deliberately a small, shared module so the synthetic fitter
(``scripts/fit_calibration.py``) and the human-gold fitter
(``scripts/fit_calibration_gold.py``) cannot drift apart.

A :class:`Pair` is one labelled observation of a reranker score. What the pair
means depends on the caller: the synthetic fitter records one positive
(query, seed chunk) per generated query and hard negatives from the same
corpus; the gold fitter records one observation per human question, labelled
``1`` when the question is answerable and scored with the **top** reranked
score, because that is the statistic the refusal decision actually uses.

Nothing here chooses a threshold for a gate. The threshold maximises balanced
accuracy on the fit set and the caller records the fit set and the held-out
number next to it.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    """One labelled reranker score and the identities that make it auditable."""

    query: str
    chunk_id: int | None
    label: int
    score: float


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def fit_platt(pairs: Sequence[Pair]) -> tuple[float, float]:
    """Fit ``p = sigmoid(a * score + b)`` by ridge-regularised Newton-Raphson.

    Scores are standardised before fitting so the slope is well conditioned,
    and a small ridge term keeps a nearly separable fit from driving ``a`` to
    infinity (which would make every probability exactly 0 or 1).
    """
    if not pairs:
        raise ValueError("calibration needs at least one pair")
    scores = [pair.score for pair in pairs]
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    scale = math.sqrt(variance) or 1.0
    a, b = 1.0, 0.0
    ridge = 1.0
    for _ in range(200):
        grad_a = grad_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for pair in pairs:
            z = (pair.score - mean) / scale
            probability = sigmoid(a * z + b)
            residual = probability - pair.label
            weight = probability * (1.0 - probability)
            grad_a += residual * z
            grad_b += residual
            h_aa += weight * z * z
            h_ab += weight * z
            h_bb += weight
        grad_a += ridge * a
        h_aa += ridge
        h_bb += ridge
        determinant = h_aa * h_bb - h_ab * h_ab
        if determinant == 0.0:
            break
        delta_a = (h_bb * grad_a - h_ab * grad_b) / determinant
        delta_b = (h_aa * grad_b - h_ab * grad_a) / determinant
        a -= delta_a
        b -= delta_b
        if abs(delta_a) < 1e-10 and abs(delta_b) < 1e-10:
            break
    return a / scale, b - a * mean / scale


def classification_rates(
    pairs: Sequence[Pair], a: float, b: float, threshold: float
) -> tuple[float, float]:
    """Return ``(true_positive_rate, true_negative_rate)`` at a probability threshold.

    A pair is "admitted" when its calibrated probability is at least the
    threshold. An empty class raises, because a rate without a denominator is
    not a measurement.
    """
    positives = [pair for pair in pairs if pair.label == 1]
    negatives = [pair for pair in pairs if pair.label == 0]
    if not positives or not negatives:
        raise ValueError("calibration needs both positive and negative pairs")
    true_positive = sum(
        1 for pair in positives if sigmoid(a * pair.score + b) >= threshold
    )
    true_negative = sum(
        1 for pair in negatives if sigmoid(a * pair.score + b) < threshold
    )
    return true_positive / len(positives), true_negative / len(negatives)


def balanced_accuracy(
    pairs: Sequence[Pair], a: float, b: float, threshold: float
) -> float:
    """Mean of TPR and TNR, the quantity the threshold maximises."""
    true_positive, true_negative = classification_rates(pairs, a, b, threshold)
    return (true_positive + true_negative) / 2.0


def choose_threshold(
    pairs: Sequence[Pair], a: float, b: float
) -> tuple[float, float]:
    """Return the balanced-accuracy-optimal threshold and its mean accuracy.

    The search runs over the midpoints between observed probabilities, so the
    returned threshold is a real decision boundary rather than a fitted value
    that happens to equal one sample. Ties prefer the tighter (higher)
    threshold.
    """
    probabilities = sorted({sigmoid(a * pair.score + b) for pair in pairs})
    candidates = [0.0, *probabilities, 1.0]
    boundaries = [
        (low + high) / 2 for low, high in itertools.pairwise(candidates)
    ]
    scored = [
        (balanced_accuracy(pairs, a, b, value), value) for value in boundaries
    ]
    accuracy, threshold = max(scored, key=lambda item: (item[0], item[1]))
    return threshold, accuracy


def canonical_fit_set(pairs: Sequence[Pair]) -> bytes:
    """Hashable form of the fit set: labels and identities, not model scores."""
    rows = [
        {"query": pair.query, "chunk_id": pair.chunk_id, "label": pair.label}
        for pair in sorted(pairs, key=lambda pair: (pair.query, str(pair.chunk_id)))
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
