"""Train a model version from the command line.

Usage:
    python -m human_or_ai.train --version v1
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from typing import Any

from human_or_ai.utils import load_config, resolve, set_seed


@dataclass
class FitResult:
    vectorizer: Any  # adapted TextVectorization layer
    model: Any  # trained Keras model (takes word ids)
    history: dict
    train_seconds: float

    def vectorize(self, raw_ds):
        """Map a (text, label) dataset to (word ids, label)."""
        import tensorflow as tf

        return raw_ds.map(lambda text, label: (self.vectorizer(tf.expand_dims(text, -1)), label))


def fit_word_model(version: str, train_df, val_df, config: dict, verbose: int = 1) -> FitResult:
    """Build the vocabulary on train_df, then train one model version with early stopping."""
    import keras

    from human_or_ai.data import make_dataset
    from human_or_ai.models import MODEL_BUILDERS, build_vectorizer

    # Reset seeds so every call is reproducible on its own.
    set_seed(config["seed"])
    bs = config["data"]["batch_size"]
    raw_train, raw_val = make_dataset(train_df, bs), make_dataset(val_df, bs)

    # 1. Build the vocabulary from the training texts only.
    vec_cfg = config["vectorizer"]
    vectorizer = build_vectorizer(vec_cfg["max_tokens"], vec_cfg["sequence_length"])
    vectorizer.adapt(raw_train.map(lambda x, y: x))
    result = FitResult(vectorizer, None, {}, 0.0)
    train_ds, val_ds = result.vectorize(raw_train), result.vectorize(raw_val)

    # 2. Build and train the model, with early stopping on the validation loss.
    model_cfg = config["model"]
    model = MODEL_BUILDERS[version](
        vec_cfg["max_tokens"], vec_cfg["sequence_length"], model_cfg["embedding_dim"], model_cfg["dropout"]
    )
    if verbose:
        model.summary()
    es_cfg = config["training"]["early_stopping"]
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=es_cfg["monitor"], patience=es_cfg["patience"], restore_best_weights=es_cfg["restore_best_weights"]
    )
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=[early_stopping],
        verbose=verbose,
    )
    result.model = model
    result.history = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    result.train_seconds = time.perf_counter() - start
    return result


def best_epoch(history: dict, monitor: str = "val_loss") -> int:
    """1-based index of the epoch with the lowest monitored value."""
    values = history[monitor]
    return int(min(range(len(values)), key=values.__getitem__)) + 1


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train a Human-or-AI detector version.")
    parser.add_argument("--version", default="v1", help="Model version to train (e.g. v1).")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml).")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    # Fix all random seeds before anything random happens.
    set_seed(config["seed"])

    import keras
    import tensorflow as tf

    from human_or_ai.data import load_splits, make_dataset
    from human_or_ai.evaluate import evaluate_model
    from human_or_ai.models import MODEL_BUILDERS, build_export_model, freeze_vectorizer

    if args.version not in MODEL_BUILDERS:
        parser.error(f"unknown version {args.version!r}; available: {sorted(MODEL_BUILDERS)}")

    # 1. Load and split the data.
    splits = load_splits(config)
    print(f"Texts before cleaning: {splits.n_raw}")
    print(f"train={len(splits.train)}  val={len(splits.val)}  test={len(splits.test)}")

    # 2. Train.
    fit = fit_word_model(args.version, splits.train, splits.val, config)

    # 3. Evaluate on the test set.
    raw_test = make_dataset(splits.test, config["data"]["batch_size"])
    threshold = config["evaluation"]["threshold"]
    test_metrics = evaluate_model(fit.model, fit.vectorize(raw_test), threshold)
    print(f"Test loss: {test_metrics['loss']:.4f}  accuracy: {test_metrics['accuracy']:.4f}")
    print("Confusion matrix:", test_metrics["confusion_matrix"])

    # 4. Save a raw-text model (vectorizer inside) and check it gives the same test score.
    export = build_export_model(freeze_vectorizer(fit.vectorizer), fit.model)
    export_loss, export_acc = export.evaluate(raw_test, verbose=0)
    out_dir = resolve(config["paths"]["artifacts"]) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    export.save(model_path)

    # 5. Write the metrics report.
    report = {
        "version": args.version,
        "seed": config["seed"],
        "data": {
            "n_raw": splits.n_raw,
            "n_train": len(splits.train),
            "n_val": len(splits.val),
            "n_test": len(splits.test),
        },
        "params": int(fit.model.count_params()),
        "epochs_run": len(fit.history["val_loss"]),
        "best_epoch": best_epoch(fit.history),
        "train_seconds": round(fit.train_seconds, 1),
        "history": fit.history,
        "test": test_metrics,
        "export_check": {"loss": float(export_loss), "accuracy": float(export_acc)},
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "platform": platform.platform(),
        },
        "model_path": str(model_path.relative_to(resolve("."))),
    }
    reports_dir = resolve(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{args.version}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {report_path}")
    return report


if __name__ == "__main__":
    main()
