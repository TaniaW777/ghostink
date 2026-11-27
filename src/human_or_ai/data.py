"""Load, merge, clean and split the two Hugging Face datasets.

With the default config, this module reproduces the v1 notebook exactly, including its known
weakness: the abstracts validation split is random, not grouped by title.

With ``data.normalize: true`` (configs/v2.yaml), texts are normalized, near-duplicates are removed,
the abstracts are split by title, and leaks from train into validation/test are removed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import pandas as pd

from human_or_ai.normalize import normalize_text

CLASS_NAMES = ["Human", "AI"]  # label 0 = human, label 1 = AI


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_raw: int  # number of texts before cleaning


def _prepare_recent(ds, split: str) -> pd.DataFrame:
    """The recent dataset uses the words "human"/"ai": map them to 0/1."""
    df = ds[split].to_pandas()[["text", "label"]]
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"human": 0, "ai": 1})
    # Keep the origin of each text for per-source and cross-dataset evaluation.
    df["source"] = "recent"
    df["title"] = None
    return df


def _prepare_abstracts(df: pd.DataFrame) -> pd.DataFrame:
    """The abstracts dataset calls its text column "abstract": rename it to "text"."""
    df = df.rename(columns={"abstract": "text"})[["text", "label", "title"]]
    # The title is kept to measure (phase 1) and later fix (phase 2) title leakage across splits.
    return df.assign(source="abstracts")[["text", "label", "source", "title"]]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without text or label, empty texts, then exact duplicates."""
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    return df.drop_duplicates(subset=["text"])


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def near_duplicate_key(text: str, n_chars: int) -> str:
    """First n_chars characters of the text, lowercased, keeping only [a-z0-9]."""
    return _NON_ALNUM_RE.sub("", str(text)[:n_chars].lower())


def drop_near_duplicates(df: pd.DataFrame, n_chars: int) -> pd.DataFrame:
    """Keep the first text of each near-duplicate key."""
    keys = df["text"].map(lambda t: near_duplicate_key(t, n_chars))
    return df[~keys.duplicated(keep="first")]


def split_abstracts_by_title(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the abstracts train set so that a title never appears on both sides."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["title"]))
    return df.iloc[train_idx], df.iloc[val_idx]


def _balance(df: pd.DataFrame) -> dict:
    """Per source: number of human and AI texts and share of the minority class."""
    out = {}
    for source in sorted(df["source"].unique()):
        part = df[df["source"] == source]
        human, ai = int((part["label"] == 0).sum()), int((part["label"] == 1).sum())
        out[source] = {"human": human, "ai": ai, "minority_share": min(human, ai) / (human + ai)}
    return out


def build_splits_v2(
    recent_parts: dict[str, pd.DataFrame],
    abs_train: pd.DataFrame,
    abs_test: pd.DataFrame,
    data_cfg: dict,
    seed: int,
) -> tuple[Splits, dict]:
    """v2 splits from prepared frames (columns text, label, source, title), plus a cleaning report.

    recent_parts has the keys "train", "validation" and "test".
    Order: normalize, drop missing/empty, drop near-duplicates, split abstracts by title,
    then remove from val/test every text whose key, or abstract whose title, is in train.
    """
    n_chars = data_cfg["near_duplicate_key_chars"]
    frames = {
        "recent/train": recent_parts["train"],
        "recent/validation": recent_parts["validation"],
        "recent/test": recent_parts["test"],
        "abstracts/train": abs_train,
        "abstracts/test": abs_test,
    }
    steps: dict[str, dict[str, int]] = {name: {"raw": len(df)} for name, df in frames.items()}
    n_raw = sum(len(df) for df in frames.values())

    for name, df in frames.items():
        # 1. Normalize every text before any deduplication.
        df = df.dropna(subset=["text", "label"])
        df = df.assign(text=df["text"].astype(str).map(normalize_text))
        df = df[df["text"] != ""]
        steps[name]["after_normalize_drop_empty"] = len(df)
        # 2. Near-duplicates inside each frame (first occurrence kept).
        df = drop_near_duplicates(df, n_chars)
        steps[name]["after_near_duplicates"] = len(df)
        frames[name] = df

    # 3. Abstracts: validation cut grouped by title.
    abs_tr, abs_val = split_abstracts_by_title(frames["abstracts/train"], data_cfg["abstracts_val_fraction"], seed)
    steps["abstracts/train"]["after_title_split_train"] = len(abs_tr)
    steps["abstracts/train"]["after_title_split_val"] = len(abs_val)

    merged = {
        "train": pd.concat([frames["recent/train"], abs_tr]),
        "val": pd.concat([frames["recent/validation"], abs_val]),
        "test": pd.concat([frames["recent/test"], frames["abstracts/test"]]),
    }
    merged_steps: dict[str, dict[str, int]] = {name: {"merged": len(df)} for name, df in merged.items()}

    # Near-duplicates across the two sources inside the same split.
    for name in merged:
        merged[name] = drop_near_duplicates(merged[name], n_chars)
        merged_steps[name]["after_near_duplicates"] = len(merged[name])

    # 4. Remove from val/test anything whose key, or abstract title, is already in train.
    train_keys = set(merged["train"]["text"].map(lambda t: near_duplicate_key(t, n_chars)))
    is_abs = merged["train"]["source"] == "abstracts"
    train_titles = set(merged["train"].loc[is_abs, "title"].dropna())
    for name in ("val", "test"):
        df = merged[name]
        keys = df["text"].map(lambda t: near_duplicate_key(t, n_chars))
        df = df[~keys.isin(train_keys)]
        merged_steps[name]["after_key_in_train_removed"] = len(df)
        title_leak = (df["source"] == "abstracts") & df["title"].isin(train_titles)
        df = df[~title_leak]
        merged_steps[name]["after_title_in_train_removed"] = len(df)
        merged[name] = df

    # Shuffle train so both datasets are mixed; keep val/test in their original order.
    train_df = merged["train"].sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = merged["val"].reset_index(drop=True)
    test_df = merged["test"].reset_index(drop=True)

    # Report only: classes are not reweighted.
    report = {
        "n_raw": n_raw,
        "near_duplicate_key_chars": n_chars,
        "counts_per_source_frame": steps,
        "counts_per_split": merged_steps,
        "balance": {"train": _balance(train_df), "val": _balance(val_df), "test": _balance(test_df)},
    }
    return Splits(train_df, val_df, test_df, n_raw), report


def load_splits(config: dict) -> Splits:
    """Download both datasets and build the train / validation / test splits."""
    from datasets import load_dataset

    seed = config["seed"]
    data_cfg = config["data"]

    # Download both datasets (cached locally after the first run).
    recent = load_dataset(data_cfg["recent_dataset"])
    abstracts = load_dataset(data_cfg["abstracts_dataset"])

    if data_cfg.get("normalize", False):
        return _load_splits_v2(config, recent, abstracts)

    # Abstracts have no validation split: shuffle their train split and cut 80 / 20.
    abs_train = abstracts["train"].to_pandas().sample(frac=1, random_state=seed).reset_index(drop=True)
    cut = int((1 - data_cfg["abstracts_val_fraction"]) * len(abs_train))

    # Merge the two datasets split by split.
    train_df = pd.concat([_prepare_recent(recent, "train"), _prepare_abstracts(abs_train[:cut])])
    val_df = pd.concat([_prepare_recent(recent, "validation"), _prepare_abstracts(abs_train[cut:])])
    test_df = pd.concat([_prepare_recent(recent, "test"), _prepare_abstracts(abstracts["test"].to_pandas())])
    n_raw = len(train_df) + len(val_df) + len(test_df)

    # Clean each split, then remove from val/test any text already seen in train (or val).
    train_df = _clean(train_df)
    val_df = _clean(val_df)
    val_df = val_df[~val_df["text"].isin(train_df["text"])]
    test_df = _clean(test_df)
    test_df = test_df[~test_df["text"].isin(train_df["text"]) & ~test_df["text"].isin(val_df["text"])]

    # Shuffle train so both datasets are mixed; keep val/test in their original order.
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return Splits(train_df, val_df, test_df, n_raw)


def _load_splits_v2(config: dict, recent, abstracts) -> Splits:
    """v2 pipeline on the downloaded datasets; writes the cleaning report to reports/data_<version>.json."""
    from human_or_ai.utils import resolve

    data_cfg = config["data"]
    if data_cfg.get("abstracts_split") != "by_title":
        raise ValueError(f"unsupported abstracts_split: {data_cfg.get('abstracts_split')!r}")
    recent_parts = {split: _prepare_recent(recent, split) for split in ("train", "validation", "test")}
    splits, report = build_splits_v2(
        recent_parts,
        _prepare_abstracts(abstracts["train"].to_pandas()),
        _prepare_abstracts(abstracts["test"].to_pandas()),
        data_cfg,
        config["seed"],
    )
    reports_dir = resolve(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"data_{config['model']['version']}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    return splits


def make_dataset(df: pd.DataFrame, batch_size: int):
    """Turn a dataframe into a batched tf.data.Dataset of (text, label)."""
    import tensorflow as tf

    texts = df["text"].astype(str).to_numpy(dtype=str)
    labels = df["label"].astype("int32").to_numpy()
    return tf.data.Dataset.from_tensor_slices((texts, labels)).batch(batch_size)
