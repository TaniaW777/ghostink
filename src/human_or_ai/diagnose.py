"""Phase 1 diagnosis of a trained model: full metrics, calibration, breakdowns, shortcuts, cross-dataset.

Usage:
    python -m human_or_ai.diagnose --version v1
    python -m human_or_ai.diagnose --version v1 --skip-cross   # no retraining, about 1 minute

Writes JSON tables to reports/diagnosis/<version>/, figures to reports/figures/,
and an auto-generated summary to reports/diagnosis/<version>/tables.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from human_or_ai.utils import load_config, resolve, set_seed


def _fmt(x, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def _md_table(df: pd.DataFrame, pct_cols=(), float_cols=()) -> str:
    """Small markdown table writer (avoids an extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in pct_cols:
                cells.append(_fmt(v, pct=True))
            elif c in float_cols or isinstance(v, float):
                cells.append(_fmt(v) if not isinstance(v, float) or abs(v) < 1000 else f"{v:.0f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summary_row(name: str, m: dict) -> dict:
    return {
        "evaluation": name,
        "n": m["n"],
        "accuracy": m["accuracy"],
        "fpr_human": m["fpr_human"],
        "tpr_ai": m["tpr_ai"],
        "roc_auc": m["roc_auc"],
        "ece": m["ece"],
    }


def predict_texts(export_model, texts, batch_size: int = 256) -> np.ndarray:
    """P(AI) for a list of raw texts."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(np.asarray(list(texts), dtype=str)).batch(batch_size)
    return export_model.predict(ds, verbose=0).ravel()


def plot_top_words(top: pd.DataFrame, path: Path, title: str) -> None:
    """Horizontal bars: most human words (blue, left) and most AI words (orange, right)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from human_or_ai.evaluate import AI_COLOR, HUMAN_COLOR, MUTED_COLOR, TEXT_COLOR

    data = top.sort_values("score")
    colors = [HUMAN_COLOR if s < 0 else AI_COLOR for s in data["score"]]
    fig, ax = plt.subplots(figsize=(7, 0.22 * len(data) + 1.2))
    ax.barh(data["word"], data["score"], color=colors, height=0.7)
    ax.axvline(0, color=TEXT_COLOR, linewidth=0.8)
    ax.set_xlabel("Word score (negative = human, positive = AI)", color=TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=11, loc="left")
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=HUMAN_COLOR, label="pushes to Human"), Patch(color=AI_COLOR, label="pushes to AI")],
        frameon=False,
        loc="lower right",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cross_dataset(splits, config: dict, version: str) -> list[dict]:
    """Train on one dataset only, test on the other one (and on its own test split for reference)."""
    from human_or_ai.data import make_dataset
    from human_or_ai.evaluate import compute_metrics, predict_dataset
    from human_or_ai.train import best_epoch, fit_word_model

    rows = []
    bs = config["data"]["batch_size"]
    threshold = config["evaluation"]["threshold"]
    for train_src, other in (("recent", "abstracts"), ("abstracts", "recent")):
        tr = splits.train[splits.train["source"] == train_src]
        va = splits.val[splits.val["source"] == train_src]
        print(f"\n[cross] training {version} on {train_src} only ({len(tr)} train / {len(va)} val)")
        fit = fit_word_model(version, tr, va, config, verbose=2)
        for test_src in (train_src, other):
            te = splits.test[splits.test["source"] == test_src]
            y, p = predict_dataset(fit.model, fit.vectorize(make_dataset(te, bs)))
            m = compute_metrics(y, p, threshold)
            rows.append(
                {
                    "train_on": train_src,
                    "test_on": test_src,
                    "kind": "in-domain" if test_src == train_src else "cross",
                    "best_epoch": best_epoch(fit.history),
                    "train_seconds": round(fit.train_seconds, 1),
                    **{k: m[k] for k in ("n", "accuracy", "fpr_human", "tpr_ai", "roc_auc", "ece")},
                    "confusion_matrix": m["confusion_matrix"],
                }
            )
            print(f"[cross] {train_src} -> {test_src}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}")
    return rows


def length_only_baseline(splits) -> dict:
    """Can the number of words alone predict the class? Logistic regression on log(word count)."""
    from sklearn.linear_model import LogisticRegression

    from human_or_ai.evaluate import compute_metrics, word_counts

    def feats(df):
        return np.log1p(word_counts(df["text"])).reshape(-1, 1)

    clf = LogisticRegression().fit(feats(splits.train), splits.train["label"])
    out = {}
    for name, part in [("all", splits.test)] + [
        (s, splits.test[splits.test["source"] == s]) for s in ("recent", "abstracts")
    ]:
        m = compute_metrics(part["label"], clf.predict_proba(feats(part))[:, 1])
        out[name] = {k: m[k] for k in ("n", "accuracy", "roc_auc", "fpr_human", "tpr_ai")}
    # Fold AUC: a length signal pointing either way is still a signal.
    for v in out.values():
        v["roc_auc_folded"] = max(v["roc_auc"], 1 - v["roc_auc"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose a trained Human-or-AI model.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-cross", action="store_true", help="Skip the two cross-dataset trainings.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(config["seed"])

    import keras

    import human_or_ai.models  # noqa: F401  registers custom_standardization for loading
    from human_or_ai import real_world, shortcuts
    from human_or_ai.data import load_splits
    from human_or_ai.evaluate import (
        compute_metrics,
        length_bins,
        metrics_by_group,
        plot_confusion,
        plot_reliability,
        word_counts,
    )

    out_dir = resolve(config["paths"]["reports"]) / "diagnosis" / args.version
    fig_dir = resolve(config["paths"]["reports"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    threshold = config["evaluation"]["threshold"]
    results: dict = {"version": args.version, "threshold": threshold}

    def save(name: str, obj) -> None:
        if isinstance(obj, pd.DataFrame):
            obj = obj.to_dict(orient="records")
        text = json.dumps(obj, indent=2, default=float)
        (out_dir / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")

    # 1. Data and saved model (no retraining).
    splits = load_splits(config)
    model_path = resolve(config["paths"]["artifacts"]) / args.version / "model.keras"
    export = keras.models.load_model(model_path, compile=False)
    test = splits.test.copy()
    test["p_ai"] = predict_texts(export, test["text"])
    test["words"] = word_counts(test["text"])

    # 2. Full metrics + calibration on the in-distribution test set.
    m = compute_metrics(test["label"], test["p_ai"], threshold)
    results["test"] = m
    save("test_metrics", m)
    plot_reliability(
        test["label"],
        test["p_ai"],
        fig_dir / f"{args.version}_reliability_test.png",
        f"{args.version}: reliability on the test set",
    )
    plot_confusion(
        m["confusion_matrix"],
        fig_dir / f"{args.version}_confusion_test.png",
        f"{args.version}: confusion matrix (test)",
    )
    print(f"test: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f} ece={m['ece']:.4f} fpr={m['fpr_human']:.4f}")

    # 3. Breakdowns by source and by length.
    by_source = metrics_by_group(test["label"], test["p_ai"], test["source"], threshold)
    test["length_bin"] = length_bins(test["text"])
    by_length = metrics_by_group(test["label"], test["p_ai"], test["length_bin"], threshold)
    by_length = by_length.set_index("group").reindex(["<50", "50-150", "150-300", ">300"]).reset_index()
    save("by_source", by_source)
    save("by_length", by_length)

    # 4. Does length alone give the answer away?
    length_only = length_only_baseline(splits)
    save("length_only_baseline", length_only)
    word_stats = (
        pd.concat([splits.train.assign(split="train"), test.assign(split="test")])
        .assign(words=lambda d: word_counts(d["text"]))
        .groupby(["source", "label"])["words"]
        .describe(percentiles=[0.5])[["count", "mean", "50%", "min", "max"]]
        .reset_index()
    )
    save("word_count_stats", word_stats)

    # 5. Shortcut patterns on the training set, per class and per source x class.
    sc_all = shortcuts.shortcut_table(splits.train)
    sc_src = shortcuts.shortcut_table(splits.train, by=["source"])
    save("shortcuts_by_class", sc_all)
    save("shortcuts_by_source", sc_src)

    # 6. Most influential words.
    top = shortcuts.top_words(export, splits.train, k=30)
    save("top_words", top)
    plot_top_words(
        top, fig_dir / f"{args.version}_top_words.png", f"{args.version}: 30 most human and 30 most AI words"
    )

    # 7. Most confident mistakes (what the model is sure about and gets wrong).
    test["confidence"] = np.where(test["p_ai"] > threshold, test["p_ai"], 1 - test["p_ai"])
    wrong = test[(test["p_ai"] > threshold).astype(int) != test["label"]]
    cols = ["source", "label", "p_ai", "words", "text"]
    worst_fp = wrong[wrong["label"] == 0].nlargest(10, "confidence")[cols]
    worst_fn = wrong[wrong["label"] == 1].nlargest(10, "confidence")[cols]
    save("worst_false_positives", worst_fp.assign(text=worst_fp["text"].str.slice(0, 400)))
    save("worst_false_negatives", worst_fn.assign(text=worst_fn["text"].str.slice(0, 400)))

    # 8. Title leakage in the abstracts dataset.
    leak = shortcuts.title_leakage(splits)
    save("title_leakage", leak)

    # 9. Cross-dataset generalization (two extra trainings).
    cross = [] if args.skip_cross else cross_dataset(splits, config, args.version)
    if cross:
        save("cross_dataset", cross)
    elif (out_dir / "cross_dataset.json").exists():
        cross = json.loads((out_dir / "cross_dataset.json").read_text(encoding="utf-8"))

    # 10. Real-world set, when it has been filled in.
    rw = real_world.load_real_world()
    rw_metrics = None
    if len(rw) and rw["label"].nunique() == 2:
        rw["p_ai"] = predict_texts(export, rw["text"])
        rw_metrics = compute_metrics(rw["label"], rw["p_ai"], threshold)
        save("real_world_metrics", rw_metrics)
        save("real_world_predictions", rw[["label_name", "source", "notes", "n_words", "p_ai"]])
        plot_reliability(
            rw["label"],
            rw["p_ai"],
            fig_dir / f"{args.version}_reliability_real_world.png",
            f"{args.version}: reliability on the real-world set",
            n_bins=5,
        )

    # 11. Auto-generated tables (the hand-written analysis lives in reports/diagnosis.md).
    summary = [_summary_row("test (in-distribution)", m)]
    summary += [_summary_row(f"cross: train {r['train_on']} -> test {r['test_on']}", r) for r in cross]
    if rw_metrics:
        summary.append(_summary_row("real world", rw_metrics))
    pct = ["accuracy", "fpr_human", "tpr_ai"]
    shortcut_cols = [
        "pattern",
        "human % texts",
        "ai % texts",
        "human mean/text",
        "ai mean/text",
        "auc_alone",
        "points_to",
    ]
    md = [
        f"# Diagnosis tables: {args.version}",
        "",
        "Auto-generated by `python -m human_or_ai.diagnose`. Do not edit by hand.",
        "",
        "## Summary",
        _md_table(pd.DataFrame(summary), pct_cols=pct),
        "",
        "## Test set: calibration",
        f"ECE = {m['ece']:.4f}, Brier = {m['brier']:.4f}, log loss = {m['log_loss']:.4f}, "
        f"mean confidence when wrong = {m['mean_confidence_when_wrong']:.3f}, "
        f"errors made with > 90% confidence = {100 * m['share_errors_above_90pct_conf']:.1f}%",
        "",
        "## By source (test)",
        _md_table(by_source, pct_cols=pct),
        "",
        "## By length in words (test)",
        _md_table(by_length, pct_cols=pct),
        "",
        "## Length alone (logistic regression on log word count)",
        _md_table(pd.DataFrame(length_only).T.reset_index(names="test subset"), pct_cols=pct),
        "",
        "## Word counts per source and class (train + test)",
        _md_table(word_stats),
        "",
        "## Shortcut patterns (train set, per class)",
        _md_table(sc_all[shortcut_cols]),
        "",
        "## Title leakage (abstracts)",
        "```json",
        json.dumps(leak, indent=2),
        "```",
        "",
        "## Top words",
        _md_table(top[["side", "word", "score", "human % texts", "ai % texts"]]),
    ]
    (out_dir / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved diagnosis to {out_dir}")


if __name__ == "__main__":
    main()
