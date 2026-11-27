"""Find surface patterns that separate the classes for the wrong reasons (dataset artifacts)."""

from __future__ import annotations

import re
import string

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Each pattern is counted on the raw text, before any cleaning.
PATTERNS: dict[str, str] = {
    # Tokenized contractions, as in HC3/Reddit human answers: "do n't", "it 's", "you 're".
    "split n't (do n't)": r"\w n't\b",
    "split 's (it 's)": r"\w 's\b",
    "split 're/'ll/'ve/'m/'d": r"\w '(?:re|ll|ve|m|d)\b",
    # Normal contractions written by people and by models.
    # "n't" needs a letter before the n, so the split form "do n't" is not counted here.
    "joined contraction (don't, it's)": r"\wn't\b|\w'(?:s|re|ll|ve|m|d)\b",
    "URL_ placeholder": r"(?i)\burl_\d+",
    "real URL (http)": r"https?://",
    "literal \\n (backslash + n)": r"\\n",
    "real line break": r"\n",
    "space before , . ; : ! ? )": r" [,.;:!?)]",
    "space after (": r"\( ",
    "straight double quote": r'"',
    "curly quotes": "[“”‘’]",
}

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_HTML_RE = re.compile(r"<[^>]+>")


def standardize(text: str) -> str:
    """Python copy of the v1 standardization: lowercase, drop HTML tags and punctuation."""
    return _PUNCT_RE.sub("", _HTML_RE.sub(" ", str(text).lower()))


def pattern_counts(texts) -> pd.DataFrame:
    """One row per text, one column per pattern (number of matches), plus word and char lengths."""
    texts = pd.Series(list(texts), dtype=str)
    out = pd.DataFrame({name: texts.str.count(rx) for name, rx in PATTERNS.items()})
    out["words"] = texts.str.split().str.len()
    out["chars"] = texts.str.len()
    return out


def shortcut_table(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Per class (and optional extra grouping): % of texts containing each pattern and mean count per text.

    Also reports the ROC-AUC of each count used alone as a classifier (0.5 = useless, 1.0 = perfect
    separation). AUC is folded so that values below 0.5 count as the same strength in the other direction.
    """
    counts = pattern_counts(df["text"])
    labels = df["label"].to_numpy()
    keys = (by or []) + ["label"]
    frame = pd.concat([df[keys].reset_index(drop=True), counts], axis=1)
    rows = []
    for name in list(PATTERNS) + ["words", "chars"]:
        row = {"pattern": name}
        for key, part in frame.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            tag = "/".join(str(k) for k in key[:-1]) + ("/" if len(key) > 1 else "")
            cls = "human" if key[-1] == 0 else "ai"
            if name in PATTERNS:
                row[f"{tag}{cls} % texts"] = 100 * float((part[name] > 0).mean())
            row[f"{tag}{cls} mean/text"] = float(part[name].mean())
        values = counts[name].to_numpy()
        auc = roc_auc_score(labels, values) if values.std() > 0 else 0.5
        row["auc_alone"] = float(max(auc, 1 - auc))
        row["points_to"] = "-" if values.std() == 0 else ("ai" if auc > 0.5 else "human")
        rows.append(row)
    return pd.DataFrame(rows)


def word_scores(export_model) -> pd.DataFrame:
    """Influence of each vocabulary word in a v1-style model: embedding vector · output weights.

    Positive = pushes towards AI, negative = pushes towards human.
    """
    vectorizer, core = export_model.layers[0], export_model.layers[1]
    embeddings = core.layers[0].get_weights()[0]
    dense = core.layers[-1].get_weights()[0]
    vocab = vectorizer.get_vocabulary()
    scores = (embeddings @ dense).ravel()
    # Skip index 0 (padding) and 1 ([UNK]).
    return pd.DataFrame({"word": vocab[2:], "score": scores[2:]})


def doc_frequency(texts, labels, words) -> pd.DataFrame:
    """Share of human and AI texts containing each word (after v1 standardization)."""
    words = list(words)
    labels = np.asarray(labels)
    token_sets = [set(standardize(t).split()) for t in texts]
    rows = []
    for w in words:
        has = np.array([w in s for s in token_sets])
        rows.append(
            {
                "word": w,
                "human % texts": 100 * float(has[labels == 0].mean()),
                "ai % texts": 100 * float(has[labels == 1].mean()),
            }
        )
    return pd.DataFrame(rows)


def top_words(export_model, train_df: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    """The k most human and k most AI words, with how often each class uses them."""
    scores = word_scores(export_model).sort_values("score")
    picked = pd.concat([scores.head(k).assign(side="human"), scores.tail(k)[::-1].assign(side="ai")])
    freq = doc_frequency(train_df["text"], train_df["label"], picked["word"])
    return picked.merge(freq, on="word").reset_index(drop=True)


def title_leakage(splits) -> dict:
    """How many abstracts in val/test share a title with an abstract in train."""

    def abstracts(df):
        return df[df["source"] == "abstracts"]

    train_titles = set(abstracts(splits.train)["title"].dropna())
    out = {}
    for name, df in (("val", splits.val), ("test", splits.test)):
        part = abstracts(df)
        shared = part["title"].isin(train_titles)
        out[name] = {
            "abstracts": int(len(part)),
            "title_also_in_train": int(shared.sum()),
            "share": float(shared.mean()) if len(part) else None,
        }
    # Titles that appear with both labels (the human and the AI version of the same paper).
    tr = abstracts(splits.train)
    both = tr.groupby("title")["label"].nunique()
    out["train_titles_with_both_labels"] = int((both > 1).sum())
    out["train_unique_titles"] = int(len(both))
    return out
