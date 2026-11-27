"""Evaluation metrics: classification, calibration and breakdowns by group.

Conventions: label 0 = human, label 1 = AI; y_prob is the predicted P(AI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

from human_or_ai.data import CLASS_NAMES

HUMAN_COLOR = "#257de0"  # blue
AI_COLOR = "#f47230"  # orange
TEXT_COLOR = "#191919"
MUTED_COLOR = "#666666"

LENGTH_BINS = [0, 50, 150, 300, np.inf]
LENGTH_LABELS = ["<50", "50-150", "150-300", ">300"]


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Group predictions into equal-width P(AI) bins; compare mean P(AI) with the real AI share."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    # Bin index 0..n_bins-1; a probability of exactly 1.0 goes in the last bin.
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append(
            {
                "bin_low": b / n_bins,
                "bin_high": (b + 1) / n_bins,
                "count": int(mask.sum()),
                "mean_prob": float(y_prob[mask].mean()) if mask.any() else np.nan,
                "frac_ai": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE: average gap between predicted P(AI) and observed AI share, weighted by bin size."""
    bins = reliability_bins(y_true, y_prob, n_bins)
    filled = bins[bins["count"] > 0]
    gaps = (filled["mean_prob"] - filled["frac_ai"]).abs()
    return float((gaps * filled["count"]).sum() / filled["count"].sum())


def compute_metrics(y_true, y_prob, threshold: float = 0.5, n_bins: int = 10) -> dict:
    """All metrics from labels and probabilities (no model needed)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_prob > threshold).astype(int)

    # Confusion matrix: rows = true class, columns = predicted class.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    # Confidence = probability given to the predicted class.
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    wrong = y_pred != y_true

    return {
        "n": int(len(y_true)),
        "n_human": int((y_true == 0).sum()),
        "n_ai": int((y_true == 1).sum()),
        "threshold": threshold,
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": {
            "tn_human_ok": int(tn),
            "fp_human_flagged_ai": int(fp),
            "fn_ai_missed": int(fn),
            "tp_ai_ok": int(tp),
        },
        # Share of human texts wrongly flagged as AI (the costly error), and share of AI texts caught.
        "fpr_human": float(fp / (fp + tn)) if (fp + tn) else None,
        "tpr_ai": float(tp / (tp + fn)) if (tp + fn) else None,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob, pos_label=1)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        # How sure the model is when it is wrong: high values = overconfident errors.
        "mean_confidence_when_wrong": float(confidence[wrong].mean()) if wrong.any() else None,
        "share_errors_above_90pct_conf": float((confidence[wrong] > 0.9).mean()) if wrong.any() else None,
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


def predict_dataset(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, P(AI)) for a batched (features, label) dataset."""
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    y_prob = model.predict(dataset, verbose=0).ravel()
    return y_true, y_prob


def evaluate_model(model, dataset, threshold: float = 0.5) -> dict:
    """Keras loss plus every metric from compute_metrics."""
    loss = model.evaluate(dataset, verbose=0, return_dict=True)["loss"]
    y_true, y_prob = predict_dataset(model, dataset)
    return {"loss": float(loss), **compute_metrics(y_true, y_prob, threshold)}


def word_counts(texts) -> np.ndarray:
    """Number of whitespace-separated words per text."""
    return np.array([len(str(t).split()) for t in texts])


def metrics_by_group(y_true, y_prob, groups, threshold: float = 0.5) -> pd.DataFrame:
    """Key metrics for each value of `groups` (e.g. source or length bin)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob), "g": np.asarray(groups)})
    rows = []
    for name, part in frame.groupby("g", sort=False, observed=True):
        m = compute_metrics(part["y"], part["p"], threshold)
        rows.append(
            {
                "group": name,
                "n": m["n"],
                "n_human": m["n_human"],
                "n_ai": m["n_ai"],
                "accuracy": m["accuracy"],
                "fpr_human": m["fpr_human"],
                "tpr_ai": m["tpr_ai"],
                "roc_auc": m["roc_auc"],
                "ece": m["ece"],
            }
        )
    return pd.DataFrame(rows)


def length_bins(texts) -> pd.Categorical:
    """Assign each text to a word-count bin: <50, 50-150, 150-300, >300."""
    return pd.cut(word_counts(texts), bins=LENGTH_BINS, labels=LENGTH_LABELS, right=False)


def plot_reliability(y_true, y_prob, path: str | Path, title: str, n_bins: int = 10) -> None:
    """Reliability diagram: predicted P(AI) vs observed AI share, with bin sizes below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = reliability_bins(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    filled = bins[bins["count"] > 0]
    centers = (bins["bin_low"] + bins["bin_high"]) / 2

    fig, (ax, ax_count) = plt.subplots(
        2, 1, figsize=(5.5, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    # Diagonal = perfect calibration.
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED_COLOR, linewidth=1, label="Perfect calibration")
    ax.plot(
        filled["mean_prob"], filled["frac_ai"], color=HUMAN_COLOR, linewidth=2, marker="o", markersize=7, label="Model"
    )
    ax.set_ylabel("Observed share of AI texts", color=TEXT_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {ece:.3f}", color=TEXT_COLOR, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#e5e5e5", linewidth=0.6)
    ax_count.bar(centers, bins["count"], width=0.9 / n_bins, color=HUMAN_COLOR)
    ax_count.set_yscale("log")
    ax_count.set_xlabel("Predicted P(AI)", color=TEXT_COLOR)
    ax_count.set_ylabel("Texts (log)", color=TEXT_COLOR)
    for a in (ax, ax_count):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED_COLOR)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: dict, path: str | Path, title: str) -> None:
    """Confusion matrix with counts and row percentages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.array([[cm["tn_human_ok"], cm["fp_human_flagged_ai"]], [cm["fn_ai_missed"], cm["tp_ai_ok"]]], dtype=float)
    row_share = grid / grid.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ink = "white" if row_share[i, j] > 0.6 else TEXT_COLOR
            ax.text(j, i, f"{int(grid[i, j])}\n{row_share[i, j]:.1%}", ha="center", va="center", color=ink)
    ax.set_xticks([0, 1], [f"Pred. {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"True {c}" for c in CLASS_NAMES])
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
