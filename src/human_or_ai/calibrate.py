"""Decision thresholds with an "Uncertain" zone.

Conventions: label 0 = human, label 1 = AI; p_ai is the predicted P(AI).
"""

from __future__ import annotations

import numpy as np

VERDICTS = ["Human", "AI", "Uncertain", "Not enough text"]


def choose_thresholds(y_true, p_ai, target_fpr: float, target_fnr: float) -> dict:
    """Pick t_ai so that at most target_fpr of human texts reach it, and t_human so that
    at most target_fnr of AI texts fall to it (both on the data given, e.g. validation)."""
    y_true = np.asarray(y_true).astype(int)
    p_ai = np.asarray(p_ai, dtype=float)
    t_ai = max(0.5, float(np.quantile(p_ai[y_true == 0], 1 - target_fpr)))
    t_human = min(0.5, float(np.quantile(p_ai[y_true == 1], target_fnr)))
    return {"t_human": t_human, "t_ai": t_ai}


def decide(p_ai: float, n_words: int, t_human: float, t_ai: float, min_words: int) -> str:
    """Verdict for one text: "Not enough text", "AI", "Human" or "Uncertain"."""
    if n_words < min_words:
        return "Not enough text"
    if p_ai >= t_ai:
        return "AI"
    if p_ai <= t_human:
        return "Human"
    return "Uncertain"
