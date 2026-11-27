"""Load and validate the hand-made "real world" test set (eval/real_world/real_world.csv).

Usage:
    python -m human_or_ai.real_world            # validate the default CSV
    python -m human_or_ai.real_world --path my.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from human_or_ai.utils import PROJECT_ROOT

REAL_WORLD_CSV = PROJECT_ROOT / "eval" / "real_world" / "real_world.csv"
COLUMNS = ["text", "label", "source", "notes"]
LABELS = {"human": 0, "ai": 1}
MIN_WORDS = 50


@dataclass
class Issue:
    row: int  # spreadsheet row number (the header is row 1)
    level: str  # "error" (row is unusable) or "warning" (row is usable but suspicious)
    message: str

    def __str__(self) -> str:
        return f"row {self.row}: {self.level}: {self.message}"


def read_csv(path: str | Path = REAL_WORLD_CSV) -> pd.DataFrame:
    """Read the CSV as text only (no type guessing, empty cells stay empty strings)."""
    try:
        # utf-8-sig also accepts files saved by Excel with a BOM.
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8; re-save it as 'CSV UTF-8'.") from exc
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; expected header: {','.join(COLUMNS)}")
    return df[COLUMNS]


def validate(df: pd.DataFrame, min_words: int = MIN_WORDS) -> list[Issue]:
    """Report empty rows, invalid labels, short texts, duplicates and missing sources."""
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(df.itertuples(index=False)):
        row = i + 2  # header is row 1
        text, label = rec.text.strip(), rec.label.strip().lower()
        if not text and not label and not rec.source.strip() and not rec.notes.strip():
            issues.append(Issue(row, "error", "empty row"))
            continue
        if not text:
            issues.append(Issue(row, "error", "empty text"))
        if label not in LABELS:
            issues.append(Issue(row, "error", f"invalid label {rec.label!r} (use 'human' or 'ai')"))
        n_words = len(text.split())
        if text and n_words < min_words:
            issues.append(Issue(row, "warning", f"only {n_words} words (< {min_words}); results are unreliable"))
        if not rec.source.strip():
            issues.append(Issue(row, "warning", "empty source"))
        if text in seen:
            issues.append(Issue(row, "warning", f"same text as row {seen[text]}"))
        elif text:
            seen[text] = row
    return issues


def load_real_world(path: str | Path = REAL_WORLD_CSV, min_words: int = 0) -> pd.DataFrame:
    """Return the usable rows with a numeric `label` (0 human, 1 AI) and a `n_words` column.

    Rows with errors are dropped. Texts shorter than `min_words` are dropped too (0 keeps all).
    """
    df = read_csv(path)
    bad_rows = {issue.row for issue in validate(df) if issue.level == "error"}
    keep = [i for i in range(len(df)) if i + 2 not in bad_rows]
    out = df.iloc[keep].copy()
    out["text"] = out["text"].str.strip()
    out["label_name"] = out["label"].str.strip().str.lower()
    out["label"] = out["label_name"].map(LABELS).astype(int)
    out["n_words"] = out["text"].str.split().str.len()
    out["paraphrased"] = out["notes"].str.lower().str.contains("paraphrased")
    return out[out["n_words"] >= min_words].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the real-world test CSV.")
    parser.add_argument("--path", default=str(REAL_WORLD_CSV))
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = parser.parse_args(argv)

    df = read_csv(args.path)
    issues = validate(df, args.min_words)
    for issue in issues:
        print(issue)
    n_errors = sum(issue.level == "error" for issue in issues)
    usable = load_real_world(args.path)
    print(f"\n{len(df)} rows, {len(usable)} usable, {n_errors} errors, {len(issues) - n_errors} warnings")
    if len(usable):
        print(usable.groupby(["label_name", "paraphrased"]).size().rename("texts").to_string())
        print(usable["source"].value_counts().rename("texts").to_string())
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
